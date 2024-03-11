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
from src.utils.utils import *
from sklearn.cluster import KMeans

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
    txt_src = open(cfg.s_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()

    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if cfg.SOURCE.TRTE == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)

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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def cal_acc_oda(cfg,loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + cfg.epsilon), dim=1) / np.log(cfg.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = cfg.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def train_source(cfg):
    cfg.output_dir_src = './source'
    if not osp.exists(cfg.output_dir_src):
        os.system('mkdir -p ' + cfg.output_dir_src)
    if not osp.exists(cfg.output_dir_src):
        os.mkdir(cfg.output_dir_src)

    dset_loaders = data_load(cfg)
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    param_group = []
    learning_rate = cfg.OPTIM.LR
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=cfg.class_num, epsilon=cfg.SOURCE.EPSILON)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(cfg.name_src, iter_num, max_iter, acc_s_te)
            logging.info(log_str)

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()
                
    torch.save(best_netF, osp.join(cfg.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(cfg.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(cfg.output_dir_src, "source_C.pt"))
    test_target(cfg)
    return netF, netB, netC

def test_target(cfg):
    ## set base network
    if cfg.MODEL.ARCH[0:3] == 'res':
        netF = network.ResBase(res_name=cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=cfg.MODEL.ARCH).cuda()  

    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()
    
    cfg.modelpath = cfg.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(cfg.modelpath))
    cfg.modelpath = cfg.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(cfg.modelpath))
    cfg.modelpath = cfg.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(cfg.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    for i in range(len(cfg.domain)):
        if i == cfg.SETTING.S:
            continue
        cfg.SETTING.T = i
        cfg.t_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
        cfg.test_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
        cfg.name = cfg.domain[cfg.SETTING.S][0].upper()+cfg.domain[cfg.SETTING.T][0].upper()
        dset_loaders = data_load(cfg)
        if cfg.DA == 'oda':
            acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(cfg.SOURCE.TRTE, cfg.name, acc_os2, acc_os1, acc_unknown)
        else:
            if cfg.SETTING.DATASET=='VISDA-C':
                acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(cfg.SOURCE.TRTE, cfg.name, acc) + '\n' + acc_list
            else:
                acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(cfg.SOURCE.TRTE, cfg.name, acc)

        logging.info(log_str)