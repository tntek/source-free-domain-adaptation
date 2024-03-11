"""
Builds upon: https://github.com/Jhyun17/CoWA-JMDS
Corresponding paper: https://proceedings.mlr.press/v162/lee22c/lee22c.pdf
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
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt
from src.utils.utils import *

matplotlib.use('Agg')

logger = logging.getLogger(__name__)

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

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # else:
    #     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
        
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        # transforms.RandomCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    # else:
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
    
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    
    return dset_loaders

def gmm(cfg,all_fea, pi, mu, all_output):    
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + cfg.COWA.EPSILON * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += cfg.COWA.EPSILON * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5*(Covi.shape[0] * np.log(2*math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma

def evaluation(loader, netF, netB, netC, cfg, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in tqdm(range(len(loader))):
        data = next(iter_test)
        inputs = data[0]
        labels = data[1].cuda()
        inputs = inputs.cuda()
        feas = netB(netF(inputs))
        outputs = netC(feas)
        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = labels.float()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
            
    _, predict = torch.max(all_output, 1)
    accuracy_return = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).data.item()

    if cfg.SETTING.DATASET=='VISDA-C':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + cfg.COWA.EPSILON2), dim=1)
    unknown_weight = 1 - ent / np.log(cfg.class_num)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg.COWA.DISTANCE == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    initc = torch.matmul(aff.t(), (all_fea))
    initc = initc / (1e-8 + aff.sum(dim=0)[:,None])

    if cfg.COWA.PICKLE and (cnt==0):
        data = {
            'all_fea': all_fea,
            'all_output': all_output,
            'all_label': all_label,
            'all_fea_orig': all_fea_orig,
        }
        filename = osp.join(cfg.output_dir, 'data_{}'.format(cfg.names[cfg.t]) + cfg.prefix + '.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('data_{}.pickle finished\n'.format(cfg.names[cfg.t]))
        
        
    ############################## Gaussian Mixture Modeling #############################

    uniform = torch.ones(len(all_fea),cfg.class_num)/cfg.class_num
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm(cfg,(all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (all_fea))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm(cfg,(all_fea), pi, mu, gamma)
        pred_label = gamma.argmax(axis=1)
            
    aff = gamma
    
    acc = (pred_label==all_label).float().mean()
    log_str = 'soft_pseudo_label_Accuracy = {:.2f}%'.format(acc * 100) + '\n'
    logging.info(log_str)
    # print(log_str)
    # cfg.out_file.write(log_str + '\n')
    # cfg.out_file.flush()

    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'

    if cfg.SETTING.DATASET=='VISDA-C':
        log_str += 'VISDA-C classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'

    logging.info(log_str)
    # cfg.out_file.write(log_str + '\n')
    # cfg.out_file.flush()
    # print(log_str)
    
    ############################## Computing JMDS score #############################

    sort_zz = zz.sort(dim=1, descending=True)[0]
    zz_sub = sort_zz[:,0] - sort_zz[:,1]
    
    LPG = zz_sub / zz_sub.max()

    if cfg.COWA.COEFF=='JMDS':
        PPL = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
        JMDS = (LPG * PPL)
    elif cfg.COWA.COEFF=='PPL':
        JMDS = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    elif cfg.COWA.COEFF=='NO':
        JMDS=torch.ones_like(LPG)
    else:
        JMDS = LPG

    sample_weight = JMDS

    if cfg.SETTING.DATASET=='VISDA-C':
        return aff, sample_weight, aacc/100
    return aff, sample_weight, accuracy
    
def KLLoss(input_, target_, coeff, cfg):
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (- target_ * torch.log(softmax + cfg.COWA.EPSILON2)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)

def mixup(x, c_batch, t_batch, netF, netB, netC, cfg):
    # weight mixup
    if cfg.COWA.ALPHA==0:
        outputs = netC(netB(netF(x)))
        return KLLoss(outputs, t_batch, c_batch, cfg)
    lam = (torch.from_numpy(np.random.beta(cfg.COWA.ALPHA, cfg.COWA.ALPHA, [len(x)]))).float().cuda()
    t_batch = t_batch.cpu()
    t_batch = torch.eye(cfg.class_num)[t_batch.argmax(dim=1)].cuda()
    shuffle_idx = torch.randperm(len(x))
    mixed_x = (lam * x.permute(1,2,3,0) + (1 - lam) * x[shuffle_idx].permute(1,2,3,0)).permute(3,0,1,2)
    mixed_c = lam * c_batch + (1 - lam) * c_batch[shuffle_idx]
    mixed_t = (lam * t_batch.permute(1,0) + (1 - lam) * t_batch[shuffle_idx].permute(1,0)).permute(1,0)
    mixed_x, mixed_c, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_c, mixed_t))
    mixed_outputs = netC(netB(netF(mixed_x)))
    return KLLoss(mixed_outputs, mixed_t, mixed_c, cfg)

def train_target(cfg):
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    ####################################################################
    modelpath = cfg.output_dir_src + '/source_F.pt'
    print('modelpath: {}'.format(modelpath))
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
        if cfg.OPTIM.LR_DECAY3 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY3}]
        else:
            v.requires_grad = False
    
    resize_size = 256
    crop_size = 224
    augment1 = transforms.Compose([
        # transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ])
            
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    cnt = 0

    dset_loaders = data_load(cfg)
    
    epochs = []
    accuracies = []
    
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        # Compute JMDS score at offline & evaluation.
        soft_pseudo_label, coeff, accuracy = evaluation(
            dset_loaders["test"], netF, netB, netC, cfg, cnt
        )
        epochs.append(cnt)
        accuracies.append(np.round(accuracy*100, 2))
    netF.train()
    netB.train()
    netC.train()
    
    uniform_ent = np.log(cfg.class_num)
    
    
    
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // (cfg.TEST.INTERVAL)
    iter_num = 0
    
    print('\nTraining start\n')
    while iter_num < max_iter:
        try:
            inputs_test, label, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, label, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue
        
        iter_num += 1
        lr_scheduler(cfg, optimizer, iter_num=iter_num, max_iter=max_iter)
        pred = soft_pseudo_label[tar_idx]
        pred_label = pred.argmax(dim=1)
        
        coeff, pred = map(torch.autograd.Variable, (coeff, pred))
        images1 = torch.autograd.Variable(augment1(inputs_test))
        images1 = images1.cuda()
        coeff = coeff.cuda()
        pred = pred.cuda()
        pred_label = pred_label.cuda()
        
        CoWA_loss = mixup(images1, coeff[tar_idx], pred, netF, netB, netC, cfg)
        
        # For warm up the start.
        if iter_num < cfg.COWA.WARM * interval_iter + 1:
            CoWA_loss *= 1e-6
            
        optimizer.zero_grad()
        CoWA_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('Evaluation iter:{}/{} start.'.format(iter_num, max_iter))
            log_str = 'Task: {}, Iter:{}/{};'.format(cfg.name, iter_num, max_iter)
            logging.info(log_str)
            # cfg.out_file.write(log_str + '\n')
            # cfg.out_file.flush()
            # print(log_str)
            
            netF.eval()
            netB.eval()
            netC.eval()
            
            cnt += 1
            with torch.no_grad():
                # Compute JMDS score at offline & evaluation.
                soft_pseudo_label, coeff, accuracy = evaluation(dset_loaders["test"], netF, netB, netC, cfg, cnt)
                epochs.append(cnt)
                accuracies.append(np.round(accuracy*100, 2))

            print('Evaluation iter:{}/{} finished.\n'.format(iter_num, max_iter))
            netF.train()
            netB.train()
            netC.train()

    ####################################################################
    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, 'ckpt_F_' + cfg.prefix + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, 'ckpt_B_' + cfg.prefix + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, 'ckpt_C_' + cfg.prefix + ".pt"))
        
        
    log_str = '\nAccuracies history : {}\n'.format(accuracies)
    # cfg.out_file.write(log_str)
    # cfg.out_file.flush()
    # print(log_str)
    logging.info(log_str)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(epochs, accuracies, 'o-')
    plt.savefig(osp.join(cfg.output_dir,'png_{}.png'.format(cfg.prefix)))
    plt.close()
    
    return netF, netB, netC

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
