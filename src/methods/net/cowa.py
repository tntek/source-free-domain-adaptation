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
from src.models import network,shot_model
import random, math
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt
from src.models.model import *
from src.data.datasets.data_loading import get_test_loader

matplotlib.use('Agg')

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(cfg, optimizer, iter_num, max_iter):
    decay = (1 + cfg.lr_gamma * iter_num / max_iter) ** (-cfg.lr_power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
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


def gmm(cfg,all_fea, pi, mu, all_output):    
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + cfg.epsilon * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += cfg.epsilon * torch.eye(temp.shape[1]).cuda() * 100
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

def evaluation(loader, base_model, cfg, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in tqdm(range(len(loader))):
        data = next(iter_test)
        inputs = data[0]
        labels = data[1].cuda()
        inputs = inputs.cuda()
        if 'image' in cfg.dset:
            feas = base_model.netF(inputs)
            if 'k' in cfg.dset:
                outputs = base_model.netC(feas)
            else:
                outputs = base_model.masking_layer(base_model.netC(feas))
        else:
            feas = base_model.encoder(inputs)
            outputs = base_model.fc(feas)
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

    if cfg.dset=='VISDA-C':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + cfg.epsilon2), dim=1)
    unknown_weight = 1 - ent / np.log(cfg.class_num)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg.distance == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    initc = torch.matmul(aff.t(), (all_fea))
    initc = initc / (1e-8 + aff.sum(dim=0)[:,None])

    if cfg.pickle and (cnt==0):
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
    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'

    if cfg.dset=='VISDA-C':
        log_str += 'VISDA-C classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'

    cfg.out_file.write(log_str + '\n')
    cfg.out_file.flush()
    print(log_str)
    
    ############################## Computing JMDS score #############################

    sort_zz = zz.sort(dim=1, descending=True)[0]
    zz_sub = sort_zz[:,0] - sort_zz[:,1]
    
    LPG = zz_sub / zz_sub.max()

    if cfg.coeff=='JMDS':
        PPL = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
        JMDS = (LPG * PPL)
    elif cfg.coeff=='PPL':
        JMDS = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    elif cfg.coeff=='NO':
        JMDS=torch.ones_like(LPG)
    else:
        JMDS = LPG

    sample_weight = JMDS

    if cfg.dset=='VISDA-C':
        return aff, sample_weight, aacc/100
    return aff, sample_weight, accuracy
    
def KLLoss(input_, target_, coeff, cfg):
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (- target_ * torch.log(softmax + cfg.epsilon2)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)

def mixup(x, c_batch, t_batch, base_model, cfg):
    # weight mixup
    if cfg.alpha==0:
        outputs = base_model(x)
        return KLLoss(outputs, t_batch, c_batch, cfg)
    lam = (torch.from_numpy(np.random.beta(cfg.alpha, cfg.alpha, [len(x)]))).float().cuda()
    t_batch = t_batch.cpu()
    t_batch = torch.eye(cfg.class_num)[t_batch.argmax(dim=1)].cuda()
    shuffle_idx = torch.randperm(len(x))
    mixed_x = (lam * x.permute(1,2,3,0) + (1 - lam) * x[shuffle_idx].permute(1,2,3,0)).permute(3,0,1,2)
    mixed_c = lam * c_batch + (1 - lam) * c_batch[shuffle_idx]
    mixed_t = (lam * t_batch.permute(1,0) + (1 - lam) * t_batch[shuffle_idx].permute(1,0)).permute(1,0)
    mixed_x, mixed_c, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_c, mixed_t))
    mixed_outputs = base_model(mixed_x)
    return KLLoss(mixed_outputs, mixed_t, mixed_c, cfg)

def train_target(cfg):
    ## set base network
    if 'image' in cfg.dset:
        if cfg.MODEL.ARCH[0:3] == 'res':
            netF = network.ResBase(res_name=cfg.MODEL.ARCH)
        elif cfg.MODEL.ARCH[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH)
        netC = network.Net2(2048,1000)
        base_model = get_model(cfg, cfg.class_num)
        netC.linear.load_state_dict(base_model.model.fc.state_dict())
        del base_model
        Shot_model = shot_model.OfficeHome_Shot(netF,netC)
        # base_model = normalize_model(Shot_model, transform.mean, transform.std)
        base_model = Shot_model
        if cfg.dset == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif cfg.dset == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif cfg.dset == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)
        elif cfg.dset == "imagenet_v":
            base_model = ImageNetXWrapper(base_model, IMAGENET_V_MASK)
    else :
        base_model = get_model(cfg, cfg.class_num)

    base_model.cuda()
    param_group = []
    for k, v in base_model.named_parameters():
        if 'netC' in k or 'fc' in k:
            v.requires_grad = False
        else:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR}]
        # else:
        #     v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)


    resize_size = 256
    crop_size = 224
    augment1 = transforms.Compose([
        # transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ])
            
    cnt = 0

    cfg.SEVERITY = [5]
    cfg.ADAPTATION = 'tent'
    cfg.NUM_EX = -1
    cfg.ALPHA_DIRICHLET = 0.0
    # dom_names_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else dom_names_all
    domain_name = cfg.type[cfg.t]
    dom_names_all = cfg.type
    target_data_loader = get_test_loader(setting=cfg.SETTING,
                                        adaptation=cfg.ADAPTATION,
                                        dataset_name=cfg.dset,
                                        root_dir=cfg.data,
                                        domain_name=domain_name,
                                        severity=cfg.level,
                                        num_examples=cfg.NUM_EX,
                                        rng_seed=cfg.seed,
                                        domain_names_all=dom_names_all,
                                        alpha_dirichlet=cfg.ALPHA_DIRICHLET,
                                        batch_size=cfg.batch_size,
                                        shuffle=True,
                                        workers=cfg.worker)

    test_data_loader = get_test_loader(setting=cfg.SETTING,
                                    adaptation=cfg.ADAPTATION,
                                    dataset_name=cfg.dset,
                                    root_dir=cfg.data,
                                    domain_name=domain_name,
                                    severity=cfg.level,
                                    num_examples=cfg.NUM_EX,
                                    rng_seed=cfg.seed,
                                    domain_names_all=dom_names_all,
                                    alpha_dirichlet=cfg.ALPHA_DIRICHLET,
                                    batch_size=cfg.batch_size*3,
                                    shuffle=False,
                                    workers=cfg.worker)

    epochs = []
    accuracies = []
    
    base_model.eval()

    with torch.no_grad():
        # Compute JMDS score at offline & evaluation.
        soft_pseudo_label, coeff, accuracy = evaluation(
            test_data_loader, base_model, cfg, cnt
        )
        epochs.append(cnt)
        accuracies.append(np.round(accuracy*100, 2))
    base_model.train()

    uniform_ent = np.log(cfg.class_num)
    
    max_iter = cfg.max_epoch * len(target_data_loader)
    interval_iter = max_iter // (cfg.interval)
    iter_num = 0
    
    print('\nTraining start\n')
    while iter_num < max_iter:
        try:
            inputs_test, label, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
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
        
        CoWA_loss = mixup(images1, coeff[tar_idx], pred, base_model, cfg)
        
        # For warm up the start.
        if iter_num < cfg.warm * interval_iter + 1:
            CoWA_loss *= 1e-6
            
        optimizer.zero_grad()
        CoWA_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('Evaluation iter:{}/{} start.'.format(iter_num, max_iter))
            log_str = 'Task: {}, Iter:{}/{};'.format(cfg.name, iter_num, max_iter)
            cfg.out_file.write(log_str + '\n')
            cfg.out_file.flush()
            print(log_str)
            
            base_model.eval()
            
            cnt += 1
            with torch.no_grad():
                # Compute JMDS score at offline & evaluation.
                soft_pseudo_label, coeff, accuracy = evaluation(test_data_loader, base_model, cfg, cnt)
                epochs.append(cnt)
                accuracies.append(np.round(accuracy*100, 2))

            print('Evaluation iter:{}/{} finished.\n'.format(iter_num, max_iter))
            base_model.train()

    ####################################################################
    if cfg.issave:   
        torch.save(base_model.state_dict(), osp.join(cfg.output_dir, 'ckpt_F_' + cfg.prefix + ".pt"))
        
    log_str = '\nAccuracies history : {}\n'.format(accuracies)
    cfg.out_file.write(log_str)
    cfg.out_file.flush()
    print(log_str)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(epochs, accuracies, 'o-')
    plt.savefig(osp.join(cfg.output_dir,'png_{}.png'.format(cfg.prefix)))
    plt.close()
    
    return base_model

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
