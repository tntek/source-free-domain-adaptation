import os, sys
sys.path.append('./')
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import network
from torch.utils.data import DataLoader
from src.utils.utils import *
from data.imagnet_prompts import imagenet_classes

logger = logging.getLogger(__name__)

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(cfg,optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = cfg.OPTIM.WD
        param_group['momentum'] = cfg.OPTIM.MOMENTUM
        param_group['nesterov'] = cfg.OPTIM.NESTEROV
    return optimizer


class ImageList_idx(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def data_load(cfg):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    # txt_src = open(cfg.s_dset_path).readlines()
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.t_dset_path).readlines()

    # dsize = len(txt_src)
    # tr_size = int(0.9*dsize)
    # print(dsize, tr_size, dsize - tr_size)
    # _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    # tr_txt = txt_src

    # dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    # dset_loaders["source_tr"] = DataLoader(dsets["source_tr"],
    #                                        batch_size=train_bs,
    #                                        shuffle=True,
    #                                        num_workers=cfg.NUM_WORKERS,
    #                                        drop_last=False)
    # dsets["source_te"] = ImageList(te_txt, transform=image_test())
    # dset_loaders["source_te"] = DataLoader(dsets["source_te"],
    #                                        batch_size=train_bs,
    #                                        shuffle=True,
    #                                        num_workers=cfg.NUM_WORKERS,
    #                                        drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=cfg.NUM_WORKERS,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs * 3,
                                      shuffle=False,
                                      num_workers=cfg.NUM_WORKERS,
                                      drop_last=False)

    return dset_loaders



def train_target(cfg):
    dset_loaders = data_load(cfg)
    ## set base network
    netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    modelpath = cfg.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        #if k.find('bn')!=-1:
        if True:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 0.1}]

    for k, v in netB.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    fea_bank=torch.randn(num_sample,512)
    score_bank = torch.randn(num_sample, cfg.class_num).cuda()

    netF.eval()
    netB.eval()
    netC.eval()


    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx=data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            output_norm=F.normalize(output)
            outputs = netC(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    acc_log=0
    while iter_num < max_iter:
        
        try:
            inputs_test, labels, tar_idx = next(iter_target)
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, labels, tar_idx = next(iter_target)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(cfg,optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(cfg,optimizer_c, iter_num=iter_num, max_iter=max_iter)


        netF.train()
        netB.train()
        netC.train()


        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_@fea_bank.T
            _, idx_near = torch.topk(distance,
                                    dim=-1,
                                    largest=True,
                                    k=cfg.NRC.K+1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]    #batch x K x C

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0],-1,-1) # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0,2,1))  # batch x K x n
            _,idx_near_near=torch.topk(distance_,dim=-1,largest=True,k=cfg.NRC.KK+1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:,:,1:] # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (
                idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    cfg.NRC.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            #weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                               cfg.class_num)  # batch x KM x C

            score_self = score_bank[tar_idx]

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, cfg.NRC.K * cfg.NRC.KK,
                                                    -1)  # batch x C x 1
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        loss = torch.mean(const)

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, cfg.NRC.K,
                                                         -1)  # batch x K x C

        loss += torch.mean((
            F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
            weight.cuda()).sum(1))

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax *
                                    torch.log(msoftmax + cfg.NRC.EPSILON))
        loss += gentropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET == 'VISDA-C':
                acc_s_te, acc_list = cal_acc_vs(dset_loaders['test'], netF, netB,
                                             netC,flag= True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%'.format(
                    cfg.name, iter_num, max_iter, acc_s_te
                ) + '\n' + 'T: ' + acc_list
            else:
                acc_s_te, _ = cal_acc_vs(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)

            logging.info(log_str)
            netF.train()
            netB.train()
            netC.train()

            if acc_s_te>acc_log:
                acc_log=acc_s_te
                torch.save(
                    netF.state_dict(),
                    osp.join(cfg.output_dir, "target_F_" + '2021_'+str(cfg.NRC.K) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(cfg.output_dir,
                                "target_B_" + '2021_' + str(cfg.NRC.K) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(cfg.output_dir,
                                "target_C_" + '2021_' + str(cfg.NRC.K) + ".pt"))

    return netF, netB, netC
