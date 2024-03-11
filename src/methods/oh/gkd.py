"""
Builds upon: https://github.com/tntek/GKD/tree/main
Corresponding paper: https://ieeexplore.ieee.org/abstract/document/9636206
"""

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import loss
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList, ImageList_idx
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import heapq
from numpy import linalg as LA
from src.utils.utils import *

logger = logging.getLogger(__name__)

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

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
#   else:
#     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
#   else:
#     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(cfg): 
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.t_dset_path).readlines()

    # if not cfg.da == 'uda':
    #     label_map_s = {}
    #     for i in range(len(cfg.src_classes)):
    #         label_map_s[cfg.src_classes[i]] = i

    #     new_tar = []
    #     for i in range(len(txt_tar)):
    #         rec = txt_tar[i]
    #         reci = rec.strip().split(' ')
    #         if int(reci[1]) in cfg.tar_classes:
    #             if int(reci[1]) in cfg.src_classes:
    #                 line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
    #                 new_tar.append(line)
    #             else:
    #                 line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
    #                 new_tar.append(line)
    #     txt_tar = new_tar.copy()
    #     txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
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

def train_target(cfg):
    dset_loaders = data_load(cfg)
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    modelpath = cfg.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if cfg.OPTIM.LR_DECAY2 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0

    iter_sw = int(max_iter/2.0)

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and cfg.GKD.CLS_PAR > 0:
            netF.eval()
            netB.eval()
            mem_label_soft, mtx_infor_nh, feas_FC = obtain_label(dset_loaders['test'], netF, netB, netC, cfg, iter_num, iter_sw)
            mem_label_soft = torch.from_numpy(mem_label_soft).cuda()
            feas_all = feas_FC[0]
            ops_all = feas_FC[1]
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)


        features_F_self = netF(inputs_test)
        features_F_nh = get_mtx_sam_wgt_nh(feas_all, mtx_infor_nh, tar_idx)
        features_F_nh = features_F_nh.cuda()
        features_F_mix = 0.8*features_F_self + 0.2*features_F_nh
        outputs_test_mix = netC(netB(features_F_mix))
        # ops_test_self = netC(netB(features_F_self))
        # outputs_test_nh = netC(netB(features_F_nh))


        if cfg.GKD.CLS_PAR > 0:
            log_probs = nn.LogSoftmax(dim=1)(outputs_test_mix)
            targets = mem_label_soft[tar_idx]
            loss_soft = (- targets * log_probs).sum(dim=1)
            classifier_loss = loss_soft.mean() 

            classifier_loss *= cfg.GKD.CLS_PAR
            if iter_num < interval_iter and cfg.SETTING.DATASET == "VISDA-C":
                classifier_loss *= 0
        else:                                                                                                                                                                                                                  
            classifier_loss = torch.tensor(0.0).cuda()


        if cfg.GKD.ENT:
            softmax_out = nn.Softmax(dim=1)(outputs_test_mix) # outputs_test_mix
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            
            if cfg.GKD.GENT:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.GKD.EPSILON))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * cfg.GKD.ENT_PAR
            classifier_loss += im_loss


        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)

            logging.info(log_str)
            netF.train()
            netB.train()

    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))
        
    return netF, netB, netC


def get_mtx_sam_wgt_nh(fea_all_f, mtx_infor_nh_f, tar_idx_f):
    mtx_idx_nh = mtx_infor_nh_f[0]
    mtx_wts_nh = mtx_infor_nh_f[1]
    idx_batch = tar_idx_f.cpu().numpy()
    fea_all_f = fea_all_f.cpu().numpy()
    ln = len(idx_batch)
    sam_wgt_nh_list = []
    for k in range(ln):
        idx_k = idx_batch[k]
        idx_nh_k = mtx_idx_nh[idx_k, 1:]
        wts_nh_k = mtx_wts_nh[idx_k, 1:][:, None]
        wts_nh_k[0] = 0.5
        wts_nh_k[1] = (0.5)*(0.5)
        wts_nh_k[2] = (0.5)*(0.5)*(0.5)
        wts_nh_k[3] = (0.5)*(0.5)*(0.5)*(0.5)
        mtx_fea_k = fea_all_f[idx_nh_k]
        mtx_fea_wgt_k = mtx_fea_k*wts_nh_k 
        sam_wgt_nh_k = np.sum(mtx_fea_wgt_k, axis=0)
        sam_wgt_nh_list.append(sam_wgt_nh_k)
    mtx_sam_wgt_nh = np.vstack(tuple(sam_wgt_nh_list))
    mtx_sam_wgt_nh_re = torch.from_numpy(mtx_sam_wgt_nh)
    return mtx_sam_wgt_nh_re


def get_mtx_output_wgt_nh(output_f, mtx_infor_nh_f, tar_idx_f):
    mtx_idx_nh = mtx_infor_nh_f[0]
    mtx_wts_nh = mtx_infor_nh_f[1]
    idx_batch = tar_idx_f.cpu().numpy()
    output_f = output_f.cpu().numpy()
    ln = len(idx_batch)
    output_wgt_nh_list = []
    for k in range(ln):
        idx_k = idx_batch[k]
        idx_nh_k = mtx_idx_nh[idx_k, 1:]
        wts_nh_k = mtx_wts_nh[idx_k, 1:][:, None]
        mtx_fea_k = output_f[idx_nh_k]
        mtx_fea_wgt_k = mtx_fea_k*wts_nh_k
        sam_wgt_nh_k = np.sum(mtx_fea_wgt_k, axis=0)
        output_wgt_nh_list.append(sam_wgt_nh_k)
    mtx_output_wgt_nh = np.vstack(tuple(output_wgt_nh_list))
    mtx_output_wgt_nh_re = torch.from_numpy(mtx_output_wgt_nh)
    return mtx_output_wgt_nh_re


def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_nearest(data_q, data_all):
    data_q_ = data_q.detach()
    data_all_ = data_all.detach()
    data_q_ = data_q_.cpu().numpy()
    data_all_ = data_all_.cpu().numpy()
    nearest_idx = get_nearest_sam_idx(data_q_, data_all_)
    re = data_all[nearest_idx, :]
    return re 

def get_nearest_sam_idx(Q, X):
    Xt = np.transpose(X)
    Simo = np.dot(Q, Xt)               
    nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
    nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
    Nor = np.dot(nq, nx)
    Sim = 1 - (Simo / Nor)
    indices_min = np.argmin(Sim, axis=1)
    indices_row = np.arange(0, Q.shape[0], 1)
    Sim[indices_row, indices_min] = 1000
    indices_min_second = np.argmin(Sim, axis=1)
    return indices_min_second



def obtain_label(loader, netF, netB, netC, cfg, num_iter_f, iter_sw_f):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas_F = netF(inputs)
            feas = netB(feas_F)
            outputs = netC(feas)
            if start_test:
                all_fea_F = feas_F.float().cpu()
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea_F = torch.cat((all_fea_F, feas_F.float().cpu()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)          # 498*256
                all_output = torch.cat((all_output, outputs.float().cpu()), 0) # 498*31
                all_label = torch.cat((all_label, labels.float()), 0)          # 498

    all_output_C = all_output
    all_output = nn.Softmax(dim=1)(all_output)

    ent = torch.sum(-all_output * torch.log(all_output + cfg.GKD.EPSILON), dim=1)
    unknown_weight = 1 - ent / np.log(cfg.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg.GKD.DISTANCE == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    xx = np.eye(K)[predict]
    cls_count = xx.sum(axis=0)
    labelset = np.where(cls_count>cfg.SHOT.THRESHOLD)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], cfg.GKD.DISTANCE)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea) 
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], cfg.GKD.DISTANCE)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy_shot = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    logging.info(log_str)

    feas_re = (all_fea_F, all_output_C)
    pred_label_new, mtx_idxnn, mtx_wts = obtain_label_nh(all_fea, pred_label, K)
    pred_label_re = pred_label_new
    mtx_re = [mtx_idxnn, mtx_wts]
    acc_knn = np.sum(pred_label_new.argmax(axis=1) == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy_ts = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc_knn * 100)
    logging.info(log_str)   
    return pred_label_re.astype('int'), mtx_re, feas_re


def obtain_all_fea(loader, netF):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
    return all_fea



def obtain_all_op(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netC(netB(netF(inputs)))
            if start_test:
                all_fea = feas.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
    return all_fea


def getListMaxNumIndex(num_list, topk):
    max_num_index = map(num_list.index, heapq.nlargest(topk,num_list))
    a = list(max_num_index)
    return a


def obtain_label_nh(feas, label_old, Kf):
    num_nn_max = 7
    VAL_MIN = -1000
    BETA = np.array(range(num_nn_max)) + 1
    ln_sam = feas.shape[0]
    idx_row = np.array(range(ln_sam))
    dd_fea = np.dot(feas, feas.T)
    oh_final = np.zeros((feas.shape[0], Kf))

    log_idx = []
    val_dd = []
    for k in range(num_nn_max):
        idx_col_max_k = dd_fea.argmax(axis=1)
        log_idx.append(idx_col_max_k)
        val_dd_k = dd_fea[idx_row, idx_col_max_k]
        val_dd.append(val_dd_k)
        dd_fea[idx_row, idx_col_max_k] = BETA[k]*VAL_MIN
    val_dd_arr = np.vstack(tuple(val_dd)).T

    oh_all = []
    for k in range(num_nn_max):
        idx_col_max_k = log_idx[k]
        lab_k = label_old[idx_col_max_k]
        one_hot_k = np.eye(Kf)[lab_k]
        wts_k = val_dd_arr[:, k][:, None]
        one_hot_w_k = one_hot_k*wts_k
        oh_final = oh_final + one_hot_w_k
        oh_all.append(oh_final)
        
    num_nn = 5
    oh_final_slt = oh_all[num_nn - 1]
    mtx_idx = np.vstack(tuple(log_idx)).T
    mtx_idx_re = mtx_idx[:, 0:num_nn]
    val_dd_re = val_dd_arr[:, 0:num_nn]
    return oh_final_slt, mtx_idx_re, val_dd_re


def get_ent(oh_final_f):
    oh_final_f = torch.from_numpy(oh_final_f)
    all_output = nn.Softmax(dim=1)(oh_final_f)
    out_ent = loss.Entropy(all_output)
    mean_ent = torch.mean(out_ent)
    out_ent_arr = out_ent.cpu().numpy()
    return mean_ent, out_ent_arr

def kennardstonealgorithm(x_variables, k):
    x_variables = np.array(x_variables)
    original_x = x_variables
    distance_to_average = ((x_variables - np.tile(x_variables.mean(axis=0), (x_variables.shape[0], 1))) ** 2).sum(axis=1)
    max_distance_sample_number = np.where(distance_to_average == np.max(distance_to_average))
    max_distance_sample_number = max_distance_sample_number[0][0]
    selected_sample_numbers = list()
    selected_sample_numbers.append(max_distance_sample_number)
    remaining_sample_numbers = np.arange(0, x_variables.shape[0], 1)
    x_variables = np.delete(x_variables, selected_sample_numbers, 0)
    remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, 0)

    for iteration in range(1, k):
        selected_samples = original_x[selected_sample_numbers, :]
        min_distance_to_selected_samples = list()
        for min_distance_calculation_number in range(0, x_variables.shape[0]):
            distance_to_selected_samples = ((selected_samples - np.tile(x_variables[min_distance_calculation_number, :],
                                                                        (selected_samples.shape[0], 1))) ** 2).sum(axis=1)
            min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
        max_distance_sample_number = np.where(
            min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
        max_distance_sample_number = max_distance_sample_number[0][0]
        selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
        x_variables = np.delete(x_variables, max_distance_sample_number, 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)
 
    return selected_sample_numbers, remaining_sample_numbers