import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from src.utils import loss,prompt_tuning,IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import  ImageList_idx,ImageList_idx_aug_fix
from sklearn.metrics import confusion_matrix
import clip
from src.utils.utils import *
from src.utils.loss import entropy

logger = logging.getLogger(__name__)

def data_load(cfg): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx_aug_fix(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
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


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_args(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target(cfg):
    clip_model, preprocess,_ = clip.load(cfg.TSD.ARCH)
    clip_model.float()
    text_inputs = clip_pre_text(cfg)
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

    loader = dset_loaders["target"]
    num_sample=len(loader.dataset)
    score_bank = torch.randn(num_sample, cfg.class_num).cuda()
    source_score_bank = torch.zeros(num_sample).cuda()
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            indx=data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs=nn.Softmax(dim=1)(outputs)
            score_bank[indx] = outputs.detach().clone()
            
            source_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean()
            source_score_bank[indx] = source_score.detach().clone()
    
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    text_features = None

    while iter_num < max_iter:
        try:
            (inputs_test, _), _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, _), _, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and cfg.TSD.CLS_PAR > 0:
            netF.eval()
            netB.eval()
            netC.eval()
            epoch_num = int(iter_num/interval_iter)
            confi_imag,confi_dis,clip_all_output, target_score_bank = obtain_label(dset_loaders['test'], netF, netB, netC,text_inputs,text_features,clip_model)
            target_score_bank = target_score_bank.cuda()
            clip_all_output = clip_all_output.cuda()
            text_features = prompt_tuning.prompt_main(cfg,confi_imag,confi_dis,iter_num)
            cfg.load = 'prompt_model.pt'
            netF.train()
            netB.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        ln_sam = softmax_out.shape[0]
        data = np.random.exponential(scale=0.1, size=ln_sam)
        data = np.expand_dims(data, axis=1)
        data = torch.from_numpy(data)

        K = softmax_out.size(1)
        _, predict = torch.max(score_bank[tar_idx], 1)
        _, clip_predict = torch.max(clip_all_output[tar_idx], 1)
        predict_one = np.eye(K)[predict.cpu()]
        clip_one = np.eye(K)[clip_predict.cpu()]

        source_score_bank[tar_idx] = 0.99*source_score_bank[tar_idx] + 0.01*target_score_bank[tar_idx]
        current_batch_scores = source_score_bank[tar_idx]
        avg_score = current_batch_scores.mean().item()

        diff = torch.abs(-torch.sum(softmax_out * torch.log(softmax_out+1e-5), 1)-avg_score)
        diff = diff.detach().clone()

        margin = 0.01*(1.01**(epoch_num-1))
        loss_ent = entropy(softmax_out , avg_score, margin)
        mask_greater = diff > margin
        indices_greater = torch.where(mask_greater)[0]

        data = data.numpy()
        predict_mix = data*predict_one + (1-data)*clip_one
        predict_mix = torch.from_numpy(predict_mix).cuda()

        if cfg.TSD.CLS_PAR > 0:
            targets = predict_mix[indices_greater]
            loss_soft = (- targets * outputs_test[indices_greater]).sum(dim=1)
            classifier_loss = loss_soft.mean()
            classifier_loss *= cfg.TSD.CLS_PAR
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        hh = clip_all_output[tar_idx]
        iic_loss = IID_losses.IID_loss(softmax_out[indices_greater], hh[indices_greater])
        classifier_loss = classifier_loss + cfg.TSD.IIC_PAR * iic_loss + cfg.TSD.LENT_PAR * loss_ent

        msoftmax = softmax_out.mean(dim=0)

        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON))
        classifier_loss = classifier_loss - cfg.TSD.GENT_PAR * gentropy_loss
        with torch.no_grad():
            score_bank[tar_idx] = softmax_out.detach().clone()

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss)

            logger.info(log_str)
            netF.train()
            netB.train()

    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))
        
    return netF, netB, netC


def print_args(cfg):
    s = "==========================================\n"    
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC,text_inputs,text_features,clip_model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            inputs_clip = data[0][1]
            labels = data[1]
            inputs = inputs.cuda()
            inputs_clip = inputs_clip.cuda() 
            feas = netB(netF(inputs)) 
            outputs = netC(feas)

            output_soft = nn.Softmax(dim=1)(outputs)
            scores = -torch.sum(output_soft * torch.log(output_soft + 1e-9), dim=1)

            if (text_features!=None):
                clip_score = clip_text(clip_model,text_features,inputs_clip)
            else :
                clip_score,_ = clip_model(inputs_clip, text_inputs)
                
            clip_score = clip_score.cpu()
            scores = scores.cpu()

            if start_test:
                all_output = outputs.float().cpu()
                all_clip_score = clip_score.float().cpu()
                all_label = labels.float().cpu()
                all_scores = scores.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                all_scores = torch.cat((all_scores, scores.float()),0)

    clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
    _, predict_clip = torch.max(clip_all_output, 1)  
    accuracy_clip = torch.sum(torch.squeeze(predict_clip).float() == all_label).item() / float(all_label.size()[0])

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    all_mix_output = (all_output+clip_all_output)/2

    confi_dis = all_mix_output.detach()
    confi_imag = loader.dataset.imgs
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    log_str = 'Accuracy = {:.2f}% -> CLIP_Accuracy  = {:.2f}%'.format(accuracy * 100, accuracy_clip * 100)
    logging.info(log_str)
    return confi_imag,confi_dis,clip_all_output, all_scores

def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.name_file) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.TSD.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts

def clip_text(model,text_features,inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits