import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import loss,prompt_tuning,IID_losses
from src.models import network,shot_model
from sklearn.metrics import confusion_matrix
import clip
from src.utils.utils import *
from src.data.datasets.data_loading import get_test_loader
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK,IMAGENET_V_MASK
from src.models.model import *
from data.imagnet_prompts import imagenet_classes
from src.utils.loss import entropy


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    # decay = 1
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


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def train_target(cfg):
    clip_model, preprocess,_ = clip.load(cfg.TSD.ARCH)
    clip_model.float()
    text_inputs = clip_pre_text(cfg)
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


    num_sample=len(target_data_loader.dataset)
    score_bank = torch.randn(num_sample, cfg.class_num).cuda()
    source_score_bank = torch.zeros(num_sample).cuda()
    base_model.eval()
    with torch.no_grad():
        iter_test = iter(target_data_loader)
        for i in range(len(target_data_loader)):
            data = next(iter_test)
            inputs = data[0]
            indx=data[-1]
            inputs = inputs.cuda()
            outputs = base_model(inputs)
            outputs=nn.Softmax(dim=1)(outputs)
            score_bank[indx] = outputs.detach().clone()

            source_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean()
            source_score_bank[indx] = source_score.detach().clone()

    max_iter = cfg.TEST.MAX_EPOCH * len(target_data_loader)
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    text_features = None

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(target_data_loader)
            inputs_test, _, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and cfg.TSD.CLS_PAR > 0:
            base_model.eval()
            epoch_num = int(iter_num/interval_iter)
            confi_imag,confi_dis,clip_all_output, target_score_bank = obtain_label(test_data_loader,base_model,text_inputs,text_features,clip_model)
            clip_all_output = clip_all_output.cuda()
            target_score_bank = target_score_bank.cuda()
            text_features = prompt_tuning.prompt_main(cfg,confi_imag,confi_dis,iter_num)
            cfg.load = 'prompt_model.pt'
            base_model.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        inputs_test = inputs_test.cuda()
        outputs_test = base_model(inputs_test)
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
        data = data.numpy()
        predict_mix = data*predict_one + (1-data)*clip_one 
        predict_mix = torch.from_numpy(predict_mix).cuda()

        source_score_bank[tar_idx] = 0.99*source_score_bank[tar_idx] + 0.01*target_score_bank[tar_idx]
        current_batch_scores = source_score_bank[tar_idx]
        avg_score = current_batch_scores.mean().item()

        diff = torch.abs(-torch.sum(softmax_out * torch.log(softmax_out+1e-5), 1)-avg_score)
        diff = diff.detach().clone()

        margin = 0.01*(1.01**(epoch_num-1))
        loss_ent = entropy(softmax_out , avg_score, margin)
        mask_greater = diff > margin
        indices_greater = torch.where(mask_greater)[0]
     
        if cfg.TSD.CLS_PAR > 0:
            targets = predict_mix[indices_greater]
            loss_soft = (- targets * outputs_test[indices_greater]).sum(dim=1)
            classifier_loss = loss_soft.mean()
            classifier_loss *= cfg.TSD.CLS_PAR
        else:
            classifier_loss = torch.tensor(0.0).cuda()
        
        hh = clip_all_output[tar_idx]

        iic_loss = IID_losses.IID_loss(softmax_out[indices_greater], hh[indices_greater])
        classifier_loss = classifier_loss + 1.0 * iic_loss

        msoftmax = softmax_out.mean(dim=0)
        classifier_loss = classifier_loss

        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.LCFD.EPSILON))
        classifier_loss = classifier_loss - cfg.TSD.GENT_PAR * gentropy_loss + cfg.TSD.LENT_PAR * loss_ent
        with torch.no_grad():
            score_bank[tar_idx] = softmax_out.detach().clone()

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_model.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(test_data_loader, base_model, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(test_data_loader, base_model, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss)

            logger.info(log_str)
            base_model.train()
        
    if cfg.ISSAVE:   
        torch.save(base_model.state_dict(), osp.join(cfg.output_dir, "target" + cfg.savename + ".pt"))

    return base_model


def obtain_label(loader, model,text_inputs,text_features,clip_model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda() 
            outputs = model(inputs)

            output_soft = nn.Softmax(dim=1)(outputs)
            scores = -torch.sum(output_soft * torch.log(output_soft + 1e-9), dim=1)

            if (text_features!=None):
                clip_score = clip_text(clip_model,text_features,inputs)
            else :
                clip_score,_ = clip_model(inputs, text_inputs)

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
    confi_imag = loader.dataset.samples
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    log_str = 'Accuracy = {:.2f}% -> CLIP_Accuracy  = {:.2f}%'.format(accuracy * 100, accuracy_clip * 100)
    logging.info(log_str)
    return confi_imag,confi_dis,clip_all_output, all_scores


def clip_pre_text(cfg):
    List_rd = []
    if 'image' in cfg.SETTING.DATASET:
        classnames_all = imagenet_classes
        classnames = []
        if cfg.SETTING.DATASET.split('_')[-1] in ['a','r','v']:
            label_mask = eval("imagenet_{}_mask".format(cfg.SETTING.DATASET.split('_')[-1]))
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

def clip_text(model,text_features,inputs_test):
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits