"""
Builds upon: https://github.com/tntek/TPDS
Corresponding paper: https://link.springer.com/article/10.1007/s11263-023-01892-w
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import IID_losses,loss
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList, ImageList_idx, ImageList_idx_aug, ImageList_idx_aug_fix
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
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
    ## prepare data
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
    # dsets["test_aug"] = ImageList_idx_aug(txt_test, transform=image_test())
    # dset_loaders["test_aug"] = DataLoader(dsets["test_aug"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.worker, drop_last=False)

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

    for k, v in netC.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False


    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    iter_num_update = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, _ = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, _ = next(iter_test)
            
        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            iter_num_update += 1
            netF.eval()
            netB.eval()
            netC.eval()
            _, feas_all, label_confi, _, _ = obtain_label_ts(dset_loaders['test'], netF, netB, netC, cfg, iter_num_update)
            netF.train()
            netB.train()
            netC.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # -----------------------------------data--------------------------------
        inputs_test = inputs_test.cuda()
        # inputs_test_aug = inputs_test_aug.cuda()
        features_test_F = netF(inputs_test)
        features_test = netB(features_test_F)
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        features_test_N, _, _ = obtain_nearest_trace(features_test_F, feas_all, label_confi)
        features_test_N = features_test_N.cuda()
        features_test_N = netB(features_test_N)
        outputs_test_N = netC(features_test_N)
        softmax_out_hyper = nn.Softmax(dim=1)(outputs_test_N)

        # -------------------------------objective------------------------------
        classifier_loss = torch.tensor(0.0).cuda()
        iic_loss = IID_losses.IID_loss(softmax_out, softmax_out_hyper)
        classifier_loss = classifier_loss + 1.0 * iic_loss

        if cfg.SETTING.DATASET == "office" or cfg.SETTING.DATASET== "domainnet126" :
            # print(1)
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.TPDS.EPSILON))
            gentropy_loss = gentropy_loss * 1.0
            classifier_loss = classifier_loss - gentropy_loss

        elif cfg.SETTING.DATASET == "office-home":
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.TPDS.EPSILON))
            gentropy_loss = gentropy_loss * 0.5
            classifier_loss = classifier_loss - gentropy_loss

        # elif cfg.dset == "office-home":
        #     msoftmax = softmax_out.mean(dim=0)
        #     gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.epsilon))
        #     gentropy_loss = gentropy_loss * 0.5
        #     classifier_loss = classifier_loss - gentropy_loss

        # --------------------------------------------------------------------    
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)

            logging.info(log_str)
            netF.train()
            netB.train()
            netC.train()


    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))
        
    return netF, netB, netC

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label_ts(loader, netF, netB, netC, cfg, iter_num_update_f):
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
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # all_logis = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + cfg.TPDS.EPSILON), dim=1)
    unknown_weight = 1 - ent / np.log(cfg.class_num)
    _, predict = torch.max(all_output, 1)

    len_unconfi = int(ent.shape[0]*0.5)
    idx_unconfi = ent.topk(len_unconfi, largest=True)[-1]
    idx_unconfi_list_ent = idx_unconfi.cpu().numpy().tolist()

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg.TPDS.DISTANCE == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>cfg.TPDS.THRESHOLD)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], cfg.TPDS.DISTANCE)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    # --------------------use dd to get confi_idx and unconfi_idx-------------
    dd_min = dd.min(axis = 1)
    dd_min_tsr = torch.from_numpy(dd_min).detach()
    dd_t_confi = dd_min_tsr.topk(int((dd.shape[0]*0.6)), largest = False)[-1]
    dd_confi_list = dd_t_confi.cpu().numpy().tolist()
    dd_confi_list.sort()
    idx_confi = dd_confi_list

    idx_all_arr = np.zeros(shape = dd.shape[0], dtype = np.int64)
    idx_all_arr[idx_confi] = 1
    idx_unconfi_arr = np.where(idx_all_arr == 0)
    idx_unconfi_list_dd = list(idx_unconfi_arr[0])

    idx_unconfi_list = list(set(idx_unconfi_list_dd).intersection(set(idx_unconfi_list_ent)))
    # ------------------------------------------------------------------------
    # idx_unconfi_list = idx_unconfi_list_dd # idx_unconfi_list_dd

    label_confi = np.ones(ent.shape[0], dtype="int64")
    label_confi[idx_unconfi_list] = 0

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = '{:.1f} AccuracyEpoch = {:.2f}% -> {:.2f}%'.format(iter_num_update_f, accuracy * 100, acc * 100)

    logging.info(log_str)

    return pred_label.astype('int'), all_fea_F, label_confi, all_label, all_output



def obtain_nearest_trace(data_q, data_all, lab_confi):
    data_q_ = data_q.detach()
    data_all_ = data_all.detach()
    data_q_ = data_q_.cpu().numpy()
    data_all_ = data_all_.cpu().numpy()
    num_sam = data_q.shape[0]
    LN_MEM = 70

    flag_is_done = 0         # indicate whether the trace process has done over the target dataset 
    ctr_oper = 0             # counter the operation time
    idx_left = np.arange(0, num_sam, 1)
    mtx_mem_rlt = -3*np.ones((num_sam, LN_MEM), dtype='int64')
    mtx_mem_ignore = np.zeros((num_sam, LN_MEM), dtype='int64')
    is_mem = 0
    mtx_log = np.zeros((num_sam, LN_MEM), dtype='int64')
    indices_row = np.arange(0, num_sam, 1)
    flag_sw_bad = 0 
    nearest_idx_last = np.array([-7])

    while flag_is_done == 0:

        nearest_idx_tmp, idx_last_tmp = get_nearest_sam_idx(data_q_, data_all_, is_mem, ctr_oper, mtx_mem_ignore, nearest_idx_last)
        is_mem = 1
        nearest_idx_last = nearest_idx_tmp

        if ctr_oper == (LN_MEM-1):    
            flag_sw_bad = 1
        else:
            flag_sw_bad = 0 

        mtx_mem_rlt[:, ctr_oper] = nearest_idx_tmp
        mtx_mem_ignore[:, ctr_oper] = idx_last_tmp
        
        lab_confi_tmp = lab_confi[nearest_idx_tmp]
        idx_done_tmp = np.where(lab_confi_tmp == 1)[0]
        idx_left[idx_done_tmp] = -1

        if flag_sw_bad == 1:
            idx_bad = np.where(idx_left >= 0)[0]
            mtx_log[idx_bad, 0] = 1
        else:
            mtx_log[:, ctr_oper] = lab_confi_tmp

        flag_len = len(np.where(idx_left >= 0)[0])
        # print("{}--the number of left:{}".format(str(ctr_oper), flag_len))
        
        if flag_len == 0 or flag_sw_bad == 1:
            # idx_nn_tmp = [list(mtx_log[k, :]).index(1) for k in range(num_sam)]
            idx_nn_step = []
            for k in range(num_sam):
                try:
                    idx_ts = list(mtx_log[k, :]).index(1)
                    idx_nn_step.append(idx_ts)
                except:
                    print("ts:", k, mtx_log[k, :])
                    # mtx_log[k, 0] = 1
                    idx_nn_step.append(0)

            idx_nn_re = mtx_mem_rlt[indices_row, idx_nn_step]
            data_re = data_all[idx_nn_re, :]
            flag_is_done = 1
        else:
            data_q_ = data_all_[nearest_idx_tmp, :]
        ctr_oper += 1

    return data_re, idx_nn_re, idx_nn_step # array



def get_nearest_sam_idx(Q, X, is_mem_f, step_num, mtx_ignore, nearest_idx_last_f): # Q、X arranged in format of row-vector
    Xt = np.transpose(X)
    Simo = np.dot(Q, Xt)               
    nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
    nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
    Nor = np.dot(nq, nx)
    Sim = 1 - (Simo / Nor)

    # Sim = cdist(Q, X, "cosine") # too slow
    # print('eeeeee \n', Sim)

    indices_min = np.argmin(Sim, axis=1)
    indices_row = np.arange(0, Q.shape[0], 1)
    
    idx_change = np.where((indices_min - nearest_idx_last_f)!=0)[0] 
    if is_mem_f == 1:
        if idx_change.shape[0] != 0:
            indices_min[idx_change] = nearest_idx_last_f[idx_change]  
    Sim[indices_row, indices_min] = 1000

    # mytst = np.eye(795)[indices_min]
    # mytst_log = np.sum(mytst, axis=0)
    # haha = np.where(mytst_log > 1)[0]
    # if haha.size != 0:
    #     print(haha)

    # Ignore the history elements. 
    if is_mem_f == 1:
        for k in range(step_num):
            indices_ingore = mtx_ignore[:, k]
            Sim[indices_row, indices_ingore] = 1000
    
    indices_min_cur = np.argmin(Sim, axis=1)
    indices_self = indices_min
    return indices_min_cur, indices_self
