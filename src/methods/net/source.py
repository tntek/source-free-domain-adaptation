import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import network,shot_model
from src.utils import loss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from src.data.datasets.data_loading import get_test_loader
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK,IMAGENET_V_MASK
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from src.models.model import *

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
            # if 'image' in args.dset:
            #     # feas = base_model.model.netF(base_model.normalize(inputs))
            #     feas = model.netF(inputs)
            #     outputs = model.masking_layer(model.netC(feas))
            # else:
            #     feas = model.encoder(inputs)
            #     outputs = model.fc(feas)

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

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def obtain_label(loader,base_model, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if 'image' in args.dset:
                # feas = base_model.model.netF(base_model.normalize(inputs))
                feas = base_model.netF(inputs)
                if 'k' in args.dset:
                    outputs = base_model.netC(feas)
                else:
                    outputs = base_model.masking_layer(base_model.netC(feas))
                # outputs = base_model(inputs)
            else:
                feas = base_model.encoder(inputs)
                outputs = base_model.fc(feas)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')




def train_target(args):
    # dset_loaders = data_load(args)
    ## set base network
    if 'image' in args.dset:
        if args.net[0:3] == 'res':
            netF = network.ResBase(res_name=args.net)
        elif args.net[0:3] == 'vgg':
            netF = network.VGGBase(vgg_name=args.net)
        netC = network.Net2(2048,1000)
        base_model,transform = get_model(args, args.class_num)
        netC.linear.load_state_dict(base_model.fc.state_dict())
        del base_model
        Shot_model = shot_model.OfficeHome_Shot(netF,netC)
        # base_model = normalize_model(Shot_model, transform.mean, transform.std)
        base_model = Shot_model
        if args.dset == "imagenet_a":
            base_model = ImageNetXWrapper(base_model, IMAGENET_A_MASK)
        elif args.dset == "imagenet_r":
            base_model = ImageNetXWrapper(base_model, IMAGENET_R_MASK)
        elif args.dset == "imagenet_d109":
            base_model = ImageNetXWrapper(base_model, IMAGENET_D109_MASK)
        elif args.dset == "imagenet_v":
            base_model = ImageNetXWrapper(base_model, IMAGENET_V_MASK)
    else :
        base_model = get_model(args, args.class_num)

    base_model.cuda()
    base_model.eval()

    for k, v in base_model.named_parameters():
        v.requires_grad = False


    args.SEVERITY = [5]
    args.ADAPTATION = 'tent'
    args.NUM_EX = -1
    args.ALPHA_DIRICHLET = 0.0
    # dom_names_loop = ["mixed"] if "mixed_domains" in args.SETTING else dom_names_all
    domain_name = args.type[args.t]
    dom_names_all = args.type
    target_data_loader = get_test_loader(setting=args.SETTING,
                                        adaptation=args.ADAPTATION,
                                        dataset_name=args.dset,
                                        root_dir=args.data,
                                        domain_name=domain_name,
                                        severity=args.level,
                                        num_examples=args.NUM_EX,
                                        rng_seed=args.seed,
                                        domain_names_all=dom_names_all,
                                        alpha_dirichlet=args.ALPHA_DIRICHLET,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        workers=args.worker)
    max_iter = args.max_epoch * len(target_data_loader)
    interval_iter = max_iter // args.interval
    iter_num = 0

    test_data_loader = get_test_loader(setting=args.SETTING,
                                    adaptation=args.ADAPTATION,
                                    dataset_name=args.dset,
                                    root_dir=args.data,
                                    domain_name=domain_name,
                                    severity=args.level,
                                    num_examples=args.NUM_EX,
                                    rng_seed=args.seed,
                                    domain_names_all=dom_names_all,
                                    alpha_dirichlet=args.ALPHA_DIRICHLET,
                                    batch_size=args.batch_size*3,
                                    shuffle=False,
                                    workers=args.worker)


    base_model.eval()
    acc_s_te, _ = cal_acc(test_data_loader,base_model,False)
    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    base_model.train()

    return base_model