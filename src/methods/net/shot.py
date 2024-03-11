"""
Builds upon: https://github.com/tim-learn/SHOT
Corresponding paper: http://proceedings.mlr.press/v119/liang20a/liang20a.pdf
"""

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import network,shot_model
from src.utils import loss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from src.data.datasets.data_loading import get_test_loader
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK,IMAGENET_V_MASK
from src.models.model import *

def cal_acc(loader, model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = model(inputs)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent



def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def obtain_label(loader,base_model, cfg):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if 'image' in cfg.SETTING.DATASET:
                feas = base_model.netF(inputs)
                if 'k' in cfg.SETTING.DATASET:
                    outputs = base_model.netC(feas)
                else:
                    outputs = base_model.masking_layer(base_model.netC(feas))
            else:
                feas = base_model.encoder(inputs)
                outputs = base_model.fc(feas)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + cfg.SHOT.EPSILON), dim=1)
    unknown_weight = 1 - ent / np.log(cfg.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg.SHOT.DISTANCE == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>cfg.SHOT.THRESHOLD)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], cfg.SHOT.DISTANCE)
    
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], cfg.SHOT.DISTANCE)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    logging.info(log_str)
    return pred_label.astype('int')




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

    param_group = []
    for k, v in base_model.named_parameters():
        if 'netC' in k or 'fc' in k:
            v.requires_grad = False
        else:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    cfg.ADAPTATION = 'tent'
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
                                    
    max_iter = cfg.TEST.MAX_EPOCH * len(target_data_loader)
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue


        if iter_num % interval_iter == 0 and cfg.SHOT.CLS_PAR > 0:
            base_model.eval()
            mem_label = obtain_label(test_data_loader, base_model, cfg)
            mem_label = torch.from_numpy(mem_label).cuda()
            base_model.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_test = base_model(inputs_test)


        if cfg.SHOT.CLS_PAR > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= cfg.SHOT.CLS_PAR
            if iter_num < interval_iter and cfg.SETTING.DATASET == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if cfg.SHOT.ENT:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if cfg.SHOT.GENT:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.SHOT.EPSILON))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * cfg.SHOT.ENT_PAR
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()
            acc_s_te, _ = cal_acc(test_data_loader,base_model,False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)
            logging.info(log_str)
            base_model.train()

    if cfg.ISSAVE:   
        torch.save(base_model.state_dict(), osp.join(cfg.output_dir, "target" + cfg.savename + ".pt"))

    return base_model