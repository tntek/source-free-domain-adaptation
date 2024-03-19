"""
Builds upon: https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation
Corresponding paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Litrico_Guiding_Pseudo-Labels_With_Uncertainty_Estimation_for_Source-Free_Unsupervised_Domain_Adaptation_CVPR_2023_paper.pdf
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import os.path as osp
import logging
from src.models.model import *
from src.utils.utils import  deepcopy_model
from sklearn.metrics import accuracy_score
from src.models import network,shot_model
from src.data.datasets.data_loading import get_test_loader
# from src.data import plue_datasets
# from torchvision import transforms

logger = logging.getLogger(__name__)
class AdaMoCo(nn.Module):
    def __init__(self, src_model, momentum_model, features_length, num_classes, dataset_length, temporal_length):
        super(AdaMoCo, self).__init__()

        self.m = 0.999

        self.first_update = True

        self.src_model = src_model
        self.momentum_model = momentum_model

        self.momentum_model.requires_grad_(False)

        self.queue_ptr = 0
        self.mem_ptr = 0

        self.T_moco = 0.07

        # queue length
        self.K = min(16384, dataset_length)
        self.memory_length = temporal_length

        self.register_buffer("features", torch.randn(features_length, self.K))
        self.register_buffer(
            "labels", torch.randint(0, num_classes, (self.K,))
        )
        self.register_buffer(
            "idxs", torch.randint(0, dataset_length, (self.K,))
        )
        self.register_buffer(
            "mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length))
        )

        self.register_buffer(
            "real_labels", torch.randint(0, num_classes, (dataset_length,))
        )

        self.features = F.normalize(self.features, dim=0)

        self.features = self.features.cuda()
        self.labels = self.labels.cuda()
        self.mem_labels = self.mem_labels.cuda()
        self.real_labels = self.real_labels.cuda()
        self.idxs = self.idxs.cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
                self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys, pseudo_labels, real_label):
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.features[:, idxs_replace] = keys.T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.real_labels[idxs_replace] = real_label
        self.queue_ptr = end % self.K

        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels
        self.mem_ptr = epoch % self.memory_length

    @torch.no_grad()
    def get_memory(self):
        return self.features, self.labels

    def forward(self, im_q, im_k=None, cls_only=False,dset=None):
        # compute query features
        # feats_q, logits_q = self.src_model(im_q, return_feats=True)
        if 'image' in dset:
            feats_q = self.src_model.netF(im_q)
            if 'k' in dset:
                logits_q = self.src_model.netC(feats_q)
            else:
                logits_q = self.src_model.masking_layer(self.src_model.netC(feats_q))
        else :
            feats_q, logits_q = self.src_model(im_q, return_feats=True)
        if cls_only:
            return feats_q, logits_q

        q = F.normalize(feats_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            if 'image' in dset:
                k = self.momentum_model.netF(im_k)
                # logits_q = self.src_model.masking_layer(self.src_model.netC(feats_q))
            else :
                k, _ = self.momentum_model(im_k, return_feats=True)
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.features.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k


def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p + 1e-5), dim=axis)


def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, num_neighbors):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)

    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    # First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


def refine_predictions(
        features,
        probs,
        banks,
        num_neighbors):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(
        features, feature_bank, probs_bank, num_neighbors
    )

    return pred_labels, probs, pred_labels_all, pred_labels_hard


def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2)
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss


@torch.no_grad()
def update_labels(banks, idxs, features, logits):
    probs = F.softmax(logits, dim=1)
    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div


def nl_criterion(output, y, num_class):
    output = torch.log(torch.clamp(1. - F.softmax(output, dim=1), min=1e-5, max=1.))

    labels_neg = ((y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1,
                                                                                      num_class).cuda()) % num_class).view(
        -1)

    l = F.nll_loss(output, labels_neg, reduction='none')

    return l


@torch.no_grad()
def eval_and_label_dataset(epoch, model, loader,cfg):
    print("Evaluating Dataset!")
    model.eval()
    logits, indices, gt_labels = [], [], []
    features = []

    for batch_idx, batch in enumerate(loader):
        imgs, targets, idxs= batch
        # imgs, targets, idxs = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        inputs = imgs[1].cuda()
        # imgs, targets, idxs= batch
        targets, idxs = targets.long().cuda(), idxs.long().cuda()
        # inputs = imgs[0].cuda()
        feats, logits_cls = model(inputs, cls_only=True,dset = cfg.SETTING.DATASET)
        features.append(feats)
        gt_labels.append(targets)
        logits.append(logits_cls)
        indices.append(idxs)

    features = torch.cat(features)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: 16384],
        "probs": probs[rand_idxs][: 16384],
        "ptr": 0,
    }

    # refine predicted labels
    pred_labels, _, _, _ = refine_predictions(features, probs, banks, cfg.PLUE.NUM_NEIGHBORS)

    acc = 100. * accuracy_score(gt_labels.to('cpu'), pred_labels.to('cpu'))

    log_str = "\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc)
    logging.info(log_str)   
    return acc, banks, gt_labels, pred_labels


def train(epoch, net, moco_model, optimizer, trainloader, banks, cfg, CE):
    loss = 0
    acc = 0

    net.train()
    moco_model.train()
    num_class = cfg.class_num
    for batch_idx, batch in enumerate(trainloader):
        imgs, y, idxs = batch
        y, idxs = y.long().cuda(), idxs.long().cuda()
        weak_x = imgs[1].cuda()
        strong_x = imgs[2].cuda()
        strong_x2 = imgs[3].cuda()

        feats_w, logits_w = moco_model(weak_x, cls_only=True,dset = cfg.SETTING.DATASET)

        if cfg.PLUE.LABEL_REFINEMENT:
            with torch.no_grad():
                probs_w = F.softmax(logits_w, dim=1)
                pseudo_labels_w, probs_w, _, _ = refine_predictions(feats_w, probs_w, banks, cfg.PLUE.LABEL_REFINEMENT)
        else:
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w = probs_w.max(1)[1]

        _, logits_q, logits_ctr, keys = moco_model(strong_x, strong_x2,dset = cfg.SETTING.DATASET)

        if cfg.PLUE.CTR:
            loss_ctr = contrastive_loss(
                logits_ins=logits_ctr,
                pseudo_labels=moco_model.mem_labels[idxs],
                mem_labels=moco_model.mem_labels[moco_model.idxs]
            )
        else:
            loss_ctr = 0

        # update key features and corresponding pseudo labels
        moco_model.update_memory(epoch, idxs, keys, pseudo_labels_w, y)

        with torch.no_grad():
            # CE weights
            max_entropy = torch.log2(torch.tensor(num_class)+ cfg.PLUE.EPSILON)
            w = entropy(probs_w)

            w = w / max_entropy
            w = torch.exp(-w+cfg.PLUE.EPSILON)

        # Standard positive learning
        if cfg.PLUE.NEG_L:
            # Standard negative learning
            loss_cls = (nl_criterion(logits_q, pseudo_labels_w, num_class)).mean()
            if cfg.PLUE.REWEIGHTING:
                loss_cls = (w * nl_criterion(logits_q, pseudo_labels_w, num_class)).mean()
        else:
            loss_cls = (CE(logits_q, pseudo_labels_w)).mean()
            if cfg.PLUE.REWEIGHTING:
                loss_cls = (w * CE(logits_q, pseudo_labels_w)).mean()

        loss_div = (div(logits_w) + div(logits_q))
        # loss_div = 0

        l = loss_cls + loss_ctr + loss_div

        update_labels(banks, idxs, feats_w, logits_w)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = 100. * accuracy_score(y.to('cpu'), logits_w.to('cpu').max(1)[1])

        loss += l.item()
        acc += accuracy

        # if batch_idx % 100 == 0:
        #     print('Epoch [%3d/%3d] Iter[%3d/%3d]\t '
        #           % (epoch, cfg.max_epoch, batch_idx + 1, len(trainloader)))
        
        #     print("Acc ", acc / (batch_idx + 1))
    f'Training acc =  {epoch}/{cfg.TEST.MAX_EPOCH} ACC {acc:.2f}%'
    log_str = "Training acc = {:.2f}".format(acc / len(trainloader))
    logging.info(log_str)

def train_target(cfg):
    ## set base network
    if 'image' in cfg.SETTING.DATASET:
        if cfg.MODEL.ARCH[0:3] == 'res':
            netF = network.ResBase(res_name=cfg.MODEL.ARCH)
        elif cfg.MODEL.ARCH[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH)
        netC = network.Net2(2048,1000)
        base_model = get_model(cfg, cfg.class_num)
        netC.linear.load_state_dict(base_model.model.fc.state_dict())
        del base_model
        Shot_model = shot_model.OfficeHome_Shot(netF,netC)
        base_model = Shot_model
        if cfg.SETTING.DATASET == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif cfg.SETTING.DATASET == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif cfg.SETTING.DATASET == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)
        elif cfg.SETTING.DATASET == "imagenet_v":
            base_model = ImageNetXWrapper(base_model, IMAGENET_V_MASK)
    else :
        base_model = get_model(cfg, cfg.class_num)
    base_model = base_model.cuda()

    base_model.cuda()
    momentun_model = deepcopy_model(base_model)
    # model = BaseModel(model, cfg.MODEL.ARCH)
    # momentun_model = BaseModel(momentun_model, cfg.MODEL.ARCH)

    optimizer = optim.SGD(base_model.parameters(), lr=cfg.OPTIM.LR, weight_decay=5e-4)

    cfg.ADAPTATION = 'adacontrast'
    domain_name = cfg.domain[cfg.SETTING.T]
    target_data_loader = get_test_loader(adaptation=cfg.ADAPTATION,
                                        dataset_name=cfg.SETTING.DATASET,
                                        root_dir=cfg.DATA_DIR,
                                        domain_name=domain_name,
                                        rng_seed=cfg.SETTING.SEED,
                                        batch_size=cfg.TEST.BATCH_SIZE,
                                        shuffle=True,
                                        workers=cfg.NUM_WORKERS)

    test_data_loader = get_test_loader(adaptation=cfg.ADAPTATION,
                                    dataset_name=cfg.SETTING.DATASET,
                                    root_dir=cfg.DATA_DIR,
                                    domain_name=domain_name,
                                    rng_seed=cfg.SETTING.SEED,
                                    batch_size=cfg.TEST.BATCH_SIZE*3,
                                    shuffle=False,
                                    workers=cfg.NUM_WORKERS)

    moco_model = AdaMoCo(src_model = base_model, momentum_model = momentun_model, features_length=cfg.bottleneck, num_classes=cfg.class_num, 
    dataset_length=len(target_data_loader.dataset), temporal_length=cfg.PLUE.TEMPORAL_LENGTH)
    CE = nn.CrossEntropyLoss(reduction='none')

    acc, banks, _, _ = eval_and_label_dataset(0, moco_model, target_data_loader, cfg)
    max_acc=0
    best_epoch=0

    for epoch in range(cfg.TEST.MAX_EPOCH + 1):
        print("Training started!")
        train(epoch, base_model, moco_model, optimizer, target_data_loader, banks, cfg, CE)  # train net1
        torch.cuda.empty_cache()
        acc, banks, gt_labels, pred_labels = eval_and_label_dataset(epoch, moco_model, test_data_loader, cfg)
        log_str = f'EPOCH: {epoch}/{cfg.TEST.MAX_EPOCH} ACC {acc:.2f}%'
        logging.info(log_str)

        if type == 'val':
            if acc > max_acc:
                max_acc = acc
                best_epoch = epoch
    if cfg.ISSAVE:   
        torch.save(base_model.state_dict(), osp.join(cfg.output_dir, "target_" + cfg.savename + ".pt"))
        
    if type == 'val':
        logger.info(f'Best epoch {best_epoch} with acc {max_acc:.2f}%')
        logging.info(log_str)
        return max_acc/100.
    else:
        return acc/100.
