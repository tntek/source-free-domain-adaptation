import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import IID_losses,miro,loss
from sklearn.metrics import confusion_matrix
import clip
import torch.nn.functional as F
from data.imagnet_prompts import imagenet_classes
from clip.custom_clip import get_coop
from copy import deepcopy
from src.models.model import get_model
from src.data.datasets.data_loading import get_test_loader
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from src.utils.utils import *
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_D109_MASK,IMAGENET_V_MASK
from src.models.model import *
logger = logging.getLogger(__name__)

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):

    # decay = (1 + gamma * iter_num / max_iter) ** (-power)
    decay = 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, base_model, flag=False):
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

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def train_target(cfg):
    text_inputs = clip_pre_text(cfg)
    base_model = get_model(cfg, cfg.class_num)
    target_logits = torch.ones(cfg.TEST.BATCH_SIZE,cfg.class_num)
    im_re_o = miro.MIRO(target_logits.shape).cuda()
    del target_logits

    model = get_coop(cfg.LCFD.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.LCFD.N_CTX, cfg.LCFD.CTX_INIT)


    for name, param in model.named_parameters():
        if "text_encoder" not in name:
            param.requires_grad_(False)

    param_group_ib = []
    param_group = []
    for k, v in base_model.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
            # if 'fc' in k:
            #     param_group += [{'params': v, 'lr': cfg.lr * cfg.lr_decay2}]
            # else:
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

    classnames = cfg.classname
    model.reset_classnames(classnames, cfg.LCFD.ARCH)
    start = True
    while iter_num < max_iter:
        try:
            inputs_test, labels,_ = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, labels,_ = next(iter_test)

        if inputs_test.size(0) == 1:
            continue
        
        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_test = base_model(inputs_test)
        outputs_test_new = outputs_test.clone().detach()
        base_model.eval()
        model.train()
        im_re_o.train()
        output_clip = test_time_adapt_eval(inputs_test, model, optimizer_ib, optim_state,cfg,outputs_test_new,im_re_o)
        output_clip = output_clip.detach().cuda().float()
        output_clip_sm = nn.Softmax(dim=1)(output_clip)
        base_model.train()
        model.eval()
        im_re_o.eval()
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
            _, clip_predict = torch.max(all_output_clip, 1)
            accuracy = torch.sum(torch.squeeze(clip_predict).float() == all_label).item() / float(all_label.size()[0])
            accuracy = accuracy*100
            log_str ='CLIP_Accuracy = {:.2f}%'.format(accuracy)
            logging.info(log_str)
            base_model.eval()
            im_re_o.eval()
            model.eval()

            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(test_data_loader, base_model, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(test_data_loader, base_model, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name, iter_num, max_iter, acc_s_te)
            logging.info(log_str)
            base_model.train()

    if cfg.ISSAVE:   
        torch.save(base_model.state_dict(), osp.join(cfg.output_dir, "target" + cfg.savename + ".pt"))
    return base_model

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


    return output


def test_time_adapt_eval(input, model, optimizer, optim_state, cfg,target_output,im_re_o):
    if cfg.LCFD.TTA_STEPS > 0:
        with torch.no_grad():
            model.train()
            im_re_o.train()
    optimizer.load_state_dict(optim_state)
    output = test_time_tuning(model, input, optimizer, cfg,target_output,im_re_o)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model.eval()
            im_re_o.eval()
            output,_ = model(input)
    output = output.cpu()
    return output

def clip_pre_text(cfg):
    List_rd = []
    if 'image' in cfg.SETTING.DATASET:
        # classnames = imagenet_classes
        classnames_all = imagenet_classes
        classnames = []
        if cfg.SETTING.DATASET.split('_')[-1] in ['a','r','v']:
            label_mask = eval("imagenet_{}_mask".format(cfg.SETTING.DATASET.split('_')[-1]))
            # classnames = [classnames_all[i] for i in label_mask]
            if 'r' in cfg.SETTING.DATASET:
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all
    else:
        with open(cfg.name_file) as f:
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
