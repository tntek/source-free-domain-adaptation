"""
Builds upon: https://github.com/Albert0147/NRC_SFDA
Corresponding paper: https://proceedings.neurips.cc/paper_files/paper/2021/file/f5deaeeae1538fb6c45901d524ee2f98-Paper.pdf
"""

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import network,shot_model
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from src.data.datasets.data_loading import get_test_loader
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_D109_MASK,IMAGENET_V_MASK
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from src.models.model import *


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer




def cal_acc(loader,base_model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = base_model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent



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
    param_group_c=[]
    for k, v in base_model.named_parameters():
        if 'netC' in k or 'fc' in k:
            param_group_c += [{'params': v, 'lr': cfg.OPTIM.LR * 1}]
        else:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

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

    num_sample=len(target_data_loader.dataset)
    fea_bank=torch.randn(num_sample,cfg.bottleneck)
    score_bank = torch.randn(num_sample, cfg.class_num).cuda()

    base_model.eval()
    with torch.no_grad():
        iter_test = iter(target_data_loader)
        for i in range(len(target_data_loader)):
            data = next(iter_test)
            inputs = data[0]
            indx=data[-1]
            inputs = inputs.cuda()
            if 'image' in cfg.SETTING.DATASET:
                output = base_model.netF(inputs)
                output_norm=F.normalize(output)
                if 'k' in cfg.SETTING.DATASET:
                    outputs = base_model.netC(output)
                else:
                    outputs = base_model.masking_layer(base_model.netC(output))
            else :
                output = base_model.encoder(inputs)
                output_norm=F.normalize(output)
                outputs = base_model.fc(output)
            outputs=nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()

    base_model.train()
    acc_log=0
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        if 'image' in cfg.SETTING.DATASET:
            features_test = output = base_model.netF(inputs_test)
            if 'k' in cfg.SETTING.DATASET:
                outputs_test = base_model.netC(features_test)
            else:
                outputs_test = base_model.masking_layer(base_model.netC(features_test))
        else :
            features_test = base_model.encoder(inputs_test)
            outputs_test = base_model.fc(features_test)

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

        # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        #loss += -torch.mean((softmax_out * score_self).sum(-1))


        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax *
                                    torch.log(msoftmax + cfg.NRC.EPSILON))
        loss += gentropy_loss

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()


        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()

            acc_s_te, _ = cal_acc(test_data_loader, base_model, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)
            logger.info(log_str)
            base_model.train()

            if acc_s_te>acc_log:
                acc_log=acc_s_te
                torch.save(
                    base_model.state_dict(),
                    osp.join(cfg.output_dir, "target_" + '2021_'+str(cfg.NRC.K) + ".pt"))

    return base_model