"""
Builds upon: https://github.com/tntek/SCLM
Corresponding paper: https://www.sciencedirect.com/science/article/abs/pii/S0893608022001897
"""
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import loss
from src.models import network,shot_model
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from src.utils.utils import *
from numpy import linalg as LA
from src.data.datasets.data_loading import get_test_loader
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK,IMAGENET_V_MASK
from src.models.model import *
import copy
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

def train_target(cfg):
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
    ent_old_val = 0

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and cfg.SCLM.CLS_PAR > 0:
            base_model.eval()
            mem_label, ent_new_val, feas_SNTg_dic, feas_SNTl_dic = obtain_label(target_data_loader, base_model, cfg, ent_old_val)
            ent_old_val = ent_new_val
            mem_label = torch.from_numpy(mem_label).cuda()
            ada_dic_num = feas_SNTg_dic.size(0)
            print("feas_SNTg_dic_num:{}".format(ada_dic_num))
            base_model.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        if 'image' in cfg.SETTING.DATASET:
            fea_f_ori = base_model.netF(inputs_test)
            # if 'k' in cfg.SETTING.DATASET:
            #     outputs = base_model.netC(feas_f)
            # else:
            #     outputs = base_model.masking_layer(base_model.netC(feas_f))
        else:
            fea_f_ori = base_model.encoder(inputs_test)
            # outputs = base_model.fc(feas_f)


        fea_f_snt_g = SNT_global_detect(fea_f_ori, feas_SNTg_dic)
        fea_f_snt_g = torch.from_numpy(fea_f_snt_g).cuda()

        fea_f_snt_l = SNT_local_detect(fea_f_ori, feas_SNTl_dic, ada_dic_num)
        fea_f_snt_l = fea_f_snt_l.cuda()

        if 'image' in cfg.SETTING.DATASET:
            if 'k' in cfg.SETTING.DATASET:
                outputs_test_ori = base_model.netC(fea_f_ori)
                outputs_test_snt_g = base_model.netC(fea_f_snt_g)
                outputs_test_snt_l = base_model.netC(fea_f_snt_l)
            else:
                outputs_test_ori = base_model.masking_layer(base_model.netC(fea_f_ori))
                outputs_test_snt_g = base_model.masking_layer(base_model.netC(fea_f_snt_g))
                outputs_test_snt_l = base_model.masking_layer(base_model.netC(fea_f_snt_l))
        else:
            outputs_test_ori = base_model.fc(fea_f_ori)
            outputs_test_snt_g = base_model.fc(fea_f_snt_g)
            outputs_test_snt_l = base_model.fc(fea_f_snt_l)


        # outputs_test_ori = netC(netB(fea_f_ori))
        # outputs_test_snt_g = netC(netB(fea_f_snt_g))
        # outputs_test_snt_l = netC(netB(fea_f_snt_l))
        
        softmax_out_ori = nn.Softmax(dim=1)(outputs_test_ori)
        softmax_out_snt_g = nn.Softmax(dim=1)(outputs_test_snt_g)
        softmax_out_snt_l = nn.Softmax(dim=1)(outputs_test_snt_l)

        output_ori_re = softmax_out_ori.unsqueeze(1)
        output_snt_g_re = softmax_out_snt_g.unsqueeze(1)
        output_snt_l_re = softmax_out_snt_l.unsqueeze(1)

        output_snt_g_re = output_snt_g_re.permute(0,2,1)
        output_snt_l_re = output_snt_l_re.permute(0,2,1)

        classifier_loss_snt_g = torch.log(torch.bmm(output_ori_re,output_snt_g_re)).sum(-1)
        classifier_loss_snt_l = torch.log(torch.bmm(output_ori_re,output_snt_l_re)).sum(-1)

        loss_const_snt_g = -torch.mean(classifier_loss_snt_g)
        loss_const_snt_l = -torch.mean(classifier_loss_snt_l)

        if cfg.SCLM.CLS_PAR > 0:
            pred = mem_label[tar_idx]
            ss_ori_loss = nn.CrossEntropyLoss()(outputs_test_ori, pred)
            ss_ori_loss *= cfg.SCLM.CLS_PAR

            ss_snt_loss = loss_const_snt_g + loss_const_snt_l
            ss_snt_loss *= cfg.SCLM.CLS_SNT

            classifier_loss = ss_ori_loss + ss_snt_loss

            if iter_num < interval_iter and cfg.SETTING.DATASET == "VISDA-C":
                classifier_loss *= 0
        
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if cfg.SCLM.ENT:
            softmax_out = softmax_out_ori + softmax_out_snt_g + softmax_out_snt_l
            entropy_loss = torch.mean(loss.Entropy(softmax_out))

            if cfg.SCLM.GENT:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.SCLM.EPSILON))
                entropy_loss -= gentropy_loss

            im_loss = entropy_loss * cfg.SCLM.ENT_PAR
            classifier_loss += im_loss
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(test_data_loader,base_model, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(test_data_loader, base_model, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)

            logging.info(log_str)
            base_model.train()

    if cfg.ISSAVE:   
        torch.save(base_model.state_dict(), osp.join(cfg.output_dir, "target_" + cfg.savename + ".pt"))
        
    return base_model


def obtain_label(loader, base_model, cfg, ent_old_val_):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if 'image' in cfg.SETTING.DATASET:
                feas_f = base_model.netF(inputs)
                if 'k' in cfg.SETTING.DATASET:
                    outputs = base_model.netC(feas_f)
                else:
                    outputs = base_model.masking_layer(base_model.netC(feas_f))
            else:
                feas_f = base_model.encoder(inputs)
                outputs = base_model.fc(feas_f)
            if start_test:
                all_fea_f = feas_f.float().cpu()
                # all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea_f = torch.cat((all_fea_f, feas_f.float().cpu()), 0)
                # all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + cfg.SCLM.EPSILON), dim=1)
    unknown_weight = 1 - ent / np.log(cfg.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    # entropy with momenum
    #===============================================================================
    SNTl_dic_ = copy.deepcopy(all_fea_f)
    ent_cur = ent.cpu().numpy()
    ent_cur_val = ent_cur
    ent_var_tmp = ent_old_val_ - ent_cur_val
    ent_var_ = np.maximum(ent_var_tmp, -ent_var_tmp)
    ent_with_mom = cfg.SCLM.NEW_ENT_PAR * ent_cur_val + (1.0 - cfg.SCLM.NEW_ENT_PAR) * ent_var_
    #===============================================================================

    if cfg.SCLM.DISTANCE == 'cosine':
        all_fea = torch.cat((all_fea_f, torch.ones(all_fea_f.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()


    # EntMomClustering
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    net_pred_label = predict.cpu().numpy()
    confi_net_pred_lst = []
    confi_all_fea_lst = []
    confi_aff_lst = []
    for i in range(K):
        idx_i = np.where(net_pred_label == i)[0]
        ent_fin_slt_cls = ent_with_mom[idx_i]
        all_fea_cls = all_fea[idx_i, :]
        aff_cls = aff[idx_i, :]
        pred_cls = net_pred_label[idx_i]
        if idx_i.shape[0] > 0:
            confi_all_fea_i, confi_aff_i, pred_i = get_confi_fea_and_op(ent_fin_slt_cls, all_fea_cls, aff_cls, pred_cls, cfg)
            confi_all_fea_lst.append(confi_all_fea_i)
            confi_aff_lst.append(confi_aff_i)
            confi_net_pred_lst.append(pred_i)

    confi_all_fea = np.vstack(tuple(confi_all_fea_lst))
    confi_aff = np.vstack(tuple(confi_aff_lst))
    confi_pred_slt = np.hstack(tuple(confi_net_pred_lst))

    initc_confi = confi_aff.transpose().dot(confi_all_fea) 
    initc_confi = initc_confi / (1e-8 + confi_aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[confi_pred_slt].sum(axis=0)
    labelset = np.where(cls_count>cfg.SCLM.THRESHOLD)
    labelset = labelset[0]

    initc_ori = aff.transpose().dot(all_fea)
    initc_ori = initc_ori / (1e-8 + aff.sum(axis=0)[:,None])

    initc = cfg.SCLM.INITC_PAR * initc_confi + (1.0 - cfg.SCLM.INITC_PAR) * initc_ori
    
    dd = cdist(all_fea, initc[labelset], cfg.SCLM.DISTANCE)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], cfg.SCLM.DISTANCE)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # construct SNTg dictionary
    #-------------------------------------------------------------------------------
    K_pred = K
    confi_cls_lst = []
    all_fea_f = all_fea_f.cpu().numpy()
    for i in range(K_pred):
        idx_i = np.where(pred_label == i)[0]
        ent_fin_slt_cls = ent_with_mom[idx_i]
        all_fea_f_cls = all_fea_f[idx_i, :]
        if idx_i.shape[0] > 0:
            confi_cls_i = SNTg_dic_cls(ent_fin_slt_cls, all_fea_f_cls, K_pred, cfg)
            confi_cls_lst.append(confi_cls_i)

    feas_SNTg_dic_ = np.vstack(tuple(confi_cls_lst))
    feas_SNTg_dic_ = torch.from_numpy(feas_SNTg_dic_).cpu()
    #-------------------------------------------------------------------------------

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    logging.info(log_str)

    return pred_label.astype('int'), ent_cur, feas_SNTg_dic_, SNTl_dic_

def get_confi_fea_and_op(ent_cls, fea_cls, aff_cls, pred_cls, cfg):
    
    len_confi = int(ent_cls.shape[0] * cfg.SCLM.CONFI_PAR) + 1
    ent_cls_tensor = torch.from_numpy(ent_cls)
    idx_confi = ent_cls_tensor.topk(len_confi, largest = False)[-1]
    fea_confi = fea_cls[idx_confi, :]
    aff_confi = aff_cls[idx_confi, :]
    pred_slt = pred_cls[idx_confi]

    return fea_confi, aff_confi, pred_slt

def SNTg_dic_cls(ent_cls, fea_cls, K, cfg):
    
    balance_num = int(2 * fea_cls.shape[1] / K)
    len_confi = int(ent_cls.shape[0] * cfg.SCLM.CONFI_PAR)
    ent_fin = torch.from_numpy(ent_cls)

    if len_confi > balance_num:
        len_confi = balance_num
        print("== cls is not balance ==")

    idx_confi = ent_fin.topk(len_confi, largest = False)[-1]
    fea_confi = fea_cls[idx_confi, :]

    return fea_confi

def SNT_global_detect(data_t_batch, data_s_confi):
    data_t = data_t_batch.detach()
    data_s = data_s_confi.detach()
    data_t_ = data_t.cpu().numpy()
    data_s_ = data_s.cpu().numpy()

    X = np.transpose(data_s_)
    Y = np.transpose(data_t_)
    beta = 20.0

    Xt = np.transpose(X)
    I = np.identity(Xt.shape[0])

    par_1 = np.matmul(Xt, X)
    par_2 = np.multiply(beta, I)

    B = par_1 + par_2
    Binv = np.linalg.inv(B)
    C = np.matmul(Binv, Xt)
    recon_fea = np.matmul(C, Y)

    idx_recon = np.argmax(recon_fea, axis = 0)
    recon_from_confi = data_s_[idx_recon, :]

    return recon_from_confi

def SNT_local_detect(data_q, data_all, ada_num_):
    data_q_ = data_q.detach()
    data_all_ = data_all.detach()
    data_q_ = data_q_.cpu().numpy()
    data_all_ = data_all_.cpu().numpy()

    sim_slt, fea_nh_nval = get_sim_in_batch(data_q_, data_all_, ada_num_)
    mask_fea_in_batch = get_mask_in_batch(sim_slt, fea_nh_nval)
    re_tmp = get_similar_fea_in_batch(mask_fea_in_batch, data_all_)

    re = torch.from_numpy(re_tmp)
    return re

def get_sim_in_batch(Q, X, basis_num_):
    Xt = np.transpose(X)
    Simo = np.dot(Q, Xt)               
    nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
    nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
    Nor = np.dot(nq, nx)
    Sim_f = 1 - (Simo / Nor) 

    indices_min = np.argmin(Sim_f, axis=1)
    indices_row = np.arange(0, Q.shape[0], 1)
    Sim_f[indices_row, indices_min] = 999
    Sim_f_sorted = np.sort(Sim_f, axis = 1)

    threshold_num = X.shape[0]//basis_num_
    get_nh_nval = Sim_f_sorted[:, threshold_num]

    return Sim_f, get_nh_nval

def get_mask_in_batch(Sim_f, fea_nh_nval):

    fea_nh_nval_f = np.expand_dims(fea_nh_nval, axis = 1)

    fea_nh_nval_zerof = np.zeros_like(Sim_f)
    fea_nh_nval_ff = fea_nh_nval_f + fea_nh_nval_zerof
    
    fea_nh_nval_slt = Sim_f - fea_nh_nval_ff

    all_1 = np.ones_like(Sim_f)
    fea_nh_nval_slt = torch.from_numpy(fea_nh_nval_slt)
    all_1 = torch.from_numpy(all_1)
    fea_nh_nval_zerof = torch.from_numpy(fea_nh_nval_zerof)

    mask_fea = torch.where(fea_nh_nval_slt <= 0.0, all_1, fea_nh_nval_zerof)
    mask_fea = mask_fea.cpu().numpy()

    return mask_fea

def get_similar_fea_in_batch(mask_fea_f, fea_all_f):
    ln = mask_fea_f.shape[0]
    ext_fea_list = []
    
    for k in range(ln):
        idx_hunter_feas = np.where(mask_fea_f[k] == 1.0)[0]
        fea_hunter_k = fea_all_f[idx_hunter_feas]

        if fea_hunter_k.shape[0] > 1:
            fea_hunter = np.mean(fea_hunter_k, axis=0)
        else:
            fea_hunter = fea_hunter_k
        ext_fea_list.append(fea_hunter) 

    ext_fea_arr = np.vstack(tuple(ext_fea_list))
    return ext_fea_arr
