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
from sklearn.metrics import confusion_matrix
from clip.custom_clip import get_coop
from src.utils import IID_losses,miro,loss
from copy import deepcopy
import torch.nn.functional as F
import clip
from src.utils.utils import *
logger = logging.getLogger(__name__)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(cfg,optimizer, iter_num, max_iter, gamma=10, power=0.75):
    if (cfg.SETTING.DATASET =='office-home'):
        # print(1)
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
    else :
        decay = 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = cfg.OPTIM.WD
        param_group['momentum'] = cfg.OPTIM.MOMENTUM
        param_group['nesterov'] = cfg.OPTIM.NESTEROV
    return optimizer

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
    text_inputs = clip_pre_text(cfg)
    dset_loaders = data_load(cfg)
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    model = get_coop(cfg.LCFD.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.LCFD.N_CTX, cfg.LCFD.CTX_INIT)
    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    modelpath = cfg.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    target_logits = torch.ones(cfg.TEST.BATCH_SIZE,cfg.class_num)
    im_re_o = miro.MIRO(target_logits.shape).cuda()
    del target_logits

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    param_group = []
    param_group_ib = []
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

    for k, v in im_re_o.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY3}]

    for k, v in model.prompt_learner.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY3}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_ib = optim.SGD(param_group_ib)
    optimizer_ib = op_copy(optimizer_ib)
    optim_state = deepcopy(optimizer_ib.state_dict())

    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    classnames = cfg.classname
    model.reset_classnames(classnames, cfg.LCFD.ARCH)
    start = True
    epoch = 0 
    while iter_num < max_iter:
        try:
            inputs_test, labels, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, labels, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue
        
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(cfg,optimizer, iter_num=iter_num, max_iter=max_iter)
        with torch.no_grad():
            outputs_test_new = netC(netB(netF(inputs_test))).detach()
        netF.eval()
        netB.eval()
        netC.eval()
        model.train()
        im_re_o.train()
        
        output_clip,_ = test_time_adapt_eval(inputs_test,labels, model, optimizer_ib, optim_state,cfg,outputs_test_new,im_re_o)
        output_clip = output_clip.detach().cuda().float()
        output_clip_sm = nn.Softmax(dim=1)(output_clip)
        netF.train()
        netB.train()
        netC.train()
        model.eval()
        im_re_o.eval()
        outputs_test = netC(netB(netF(inputs_test)))
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        
        if (cfg.LCFD.LOSS_FUNC=="l1"):
            loss_l1 = torch.nn.L1Loss(reduction='mean')
            classifier_loss = loss_l1(softmax_out, output_clip_sm)
            classifier_loss *= cfg.LCFD.CLS_PAR
        elif (cfg.LCFD.LOSS_FUNC=="l2"):
            loss_l2 = torch.nn.MSELoss(reduction='mean')
            classifier_loss = loss_l2(softmax_out,output_clip_sm)
            classifier_loss *= cfg.LCFD.CLS_PAR
        elif (cfg.LCFD.LOSS_FUNC=="iid"):
            classifier_loss = IID_losses.IID_loss(softmax_out,output_clip_sm)
            classifier_loss *= cfg.LCFD.CLS_PAR
        elif (cfg.LCFD.LOSS_FUNC=="kl"):
            classifier_loss = F.kl_div(softmax_out.log(),output_clip_sm, reduction='sum')
            classifier_loss *= cfg.LCFD.CLS_PAR
        elif (cfg.LCFD.LOSS_FUNC=="sce"):
            _, pred = torch.max(output_clip, 1)
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= cfg.LCFD.CLS_PAR
        else :
            classifier_loss = torch.tensor(0.0).cuda()


        if cfg.LCFD.ENT:
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if cfg.LCFD.GENT:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON))
                entropy_loss -= cfg.LCFD.GENT_PAR*gentropy_loss
            im_loss = entropy_loss
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        with torch.no_grad():
            if start:
                all_output_clip = output_clip.float().cpu()
                all_label = labels.float()
                start = False
            else:
                all_output_clip = torch.cat((all_output_clip, output_clip.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            start = True
            epoch = epoch + 1
            _, clip_predict = torch.max(all_output_clip, 1)
            accuracy = torch.sum(torch.squeeze(clip_predict).float() == all_label).item() / float(all_label.size()[0])
            accuracy = accuracy*100
            log_str ='CLIP_Accuracy = {:.2f}%'.format(accuracy)
            logging.info(log_str)
            netF.eval()
            netB.eval()
            netC.eval()
            im_re_o.eval()
            model.eval()

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

def test_time_tuning(model, inputs, optimizer, cfg,target_output,im_re_o):

    target_output = target_output.cuda()

    for j in range(cfg.LCFD.TTA_STEPS):
        with torch.cuda.amp.autocast():

            output_logits,_ = model(inputs)             
            if(output_logits.shape[0]!=cfg.TEST.BATCH_SIZE):
                padding_f=torch.zeros([cfg.TEST.BATCH_SIZE-output_logits.shape[0],output_logits.shape[1]],dtype=torch.float).cuda()
                output_logits = torch.cat((output_logits, padding_f.float()), 0)
                target_output =  torch.cat((target_output, padding_f.float()), 0)

            im_loss_o, Delta = im_re_o.update(output_logits,target_output)
            Delta = 1.0/(Delta+1e-5)
            Delta = nn.Softmax(dim=1)(Delta)
            output_logits_sm = nn.Softmax(dim=1)(output_logits)
            output = Delta*output_logits_sm
            iic_loss = IID_losses.IID_loss(output, output_logits_sm)
            loss = 0.5*(iic_loss - 0.0003*im_loss_o) #0.0003
            if(inputs.shape[0]!=cfg.TEST.BATCH_SIZE):
                output = output[:inputs.shape[0]]
                target_output = target_output[:inputs.shape[0]]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return output,loss


def test_time_adapt_eval(input,target, model, optimizer, optim_state, cfg,target_output,im_re_o):
    if cfg.LCFD.TTA_STEPS > 0:
        with torch.no_grad():
            model.train()
            im_re_o.train()
    optimizer.load_state_dict(optim_state)
    output,loss_ib = test_time_tuning(model, input, optimizer, cfg,target_output,im_re_o)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model.eval()
            im_re_o.eval()
            output,_ = model(input)
    output = output.cpu()
    return output,loss_ib


def clip_pre_text(cfg):
    List_rd = []
    with open(cfg.LCFD.NAME_FILE) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    cfg.classname = classnames
    prompt_prefix = cfg.LCFD.CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts