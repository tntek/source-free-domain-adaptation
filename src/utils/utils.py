import argparse
import logging
import os
import random
import sys
from copy import deepcopy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import torch.distributed as dist

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def adjust_learning_rate(optimizer, progress, args):
    """
    Decay the learning rate based on epoch or iteration.
    """
    if args.optim_cos:
        decay = 0.5 * (1.0 + math.cos(math.pi * progress / args.full_progress))
    elif args.optim_exp:
        decay = (1 + 10 * progress / args.full_progress) ** -0.75
    else:
        decay = 1.0
        for milestone in args.schedule:
            decay *= args.gamma if progress >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay

    return decay

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def mean(items):
    return sum(items) / len(items)


def max_with_index(values):
    best_v = values[0]
    best_i = 0
    for i, v in enumerate(values):
        if v > best_v:
            best_v = v
            best_i = i
    return best_v, best_i


def shuffle(*items):
    example, *_ = items
    batch_size, *_ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]


def to_device(*items):
    return [item.to(device=device) for item in items]


def set_reproducible(seed=0):
    '''
    To ensure the reproducibility, refer to https://pytorch.org/docs/stable/notes/randomness.html.
    Note that completely reproducible results are not guaranteed.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger


def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1


def compute_flops(module: nn.Module, size, skip_pattern, device):
    # print(module._auxiliary)
    def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        *_, h, w = output.shape
        module.output_size = (h, w)

    hooks = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("init hool for", name)
            hooks.append(m.register_forward_hook(size_hook))
    with torch.no_grad():
        training = module.training
        module.eval()
        module(torch.rand(size).to(device))
        module.train(mode=training)
        # print(f"training={training}")
    for hook in hooks:
        hook.remove()

    flops = 0
    for name, m in module.named_modules():
        if skip_pattern in name:
            continue
        if isinstance(m, nn.Conv2d):
            # print(name)
            h, w = m.output_size
            kh, kw = m.kernel_size
            flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
        if isinstance(module, nn.Linear):
            flops += m.in_features * m.out_features
    return flops


def compute_nparam(module: nn.Module, skip_pattern):
    n_param = 0
    for name, p in module.named_parameters():
        if skip_pattern not in name:
            n_param += p.numel()
    return n_param


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            predictions = output.argmax(1)
            correct += (predictions == labels.to(device)).float().sum()
            pass

    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy


def split_up_model(model):
    modules = list(model.children())[:-1]
    classifier = list(model.children())[-1]
    while not isinstance(classifier, nn.Linear):
        sub_modules = list(classifier.children())[:-1]
        modules.extend(sub_modules)
        classifier = list(classifier.children())[-1]
    featurizer = nn.Sequential(*modules)

    return featurizer, classifier


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def get_output(encoder, classifier, x,arch):
    x = encoder(x)
    if arch=='WideResNet':
        x = F.avg_pool2d(x, 8)
        features = x.view(-1, classifier.in_features)
    elif arch=='vit':
        features = x[:, 0]
    else:
        features = x.squeeze()
    return features, classifier(features)


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def cal_acc(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs=model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0])

    return accuracy * 100

def cal_acc_vs(loader, netF, netB, netC, flag=False):
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
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent



def del_wn_hook(model):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                delattr(module, hook.name)


def restore_wn_hook(model, name='weight'):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook(module, name)


def deepcopy_model(model):
    ### del weight norm hook
    del_wn_hook(model)

    ### copy model
    model_cp = deepcopy(model)

    ### restore weight norm hook
    restore_wn_hook(model)
    restore_wn_hook(model_cp)

    return model_cp


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--cfg', default=None, type=str, help='path to config file')
    parser.add_argument('--MODEL_CONTINUAL', default=None, type=str)
    parser.add_argument('--OPTIM_LR', default=None, type=float)
    parser.add_argument('--BN_ALPHA', default=None, type=float)
    parser.add_argument('--output_dir', default=None, type=str, help='path to output_dir file')
    parser.add_argument('--COTTA_RST', default=None, type=float)
    parser.add_argument('--COTTA_AP', default=None, type=float)
    parser.add_argument('--M_TEACHER_MOMENTUM', default=None, type=float)
    parser.add_argument('--EATA_DM', default=None, type=float)
    parser.add_argument('--EATA_FISHER_ALPHA', default=None, type=float)
    parser.add_argument('--T3A_FILTER_K', default=None, type=int)
    parser.add_argument('--LAME_AFFINITY', default=None, type=str)
    parser.add_argument('--LAME_KNN', default=None, type=int)

    parser.add_argument('--TEST_EPOCH', default=None, type=int)
    parser.add_argument('--SHOT_CLS_PAR', default=None, type=float)
    parser.add_argument('--SHOT_ENT_PAR', default=None, type=float)
    parser.add_argument('--NRC_K', default=None, type=int)
    parser.add_argument('--NRC_KK', default=None, type=int)
    parser.add_argument('--SAR_RESET_CONSTANT', default=None, type=float)
    parser.add_argument('--PLUE_NUM_NEIGHBORS', default=None, type=int)
    parser.add_argument('--ADACONTRAST_NUM_NEIGHBORS', default=None, type=int)
    parser.add_argument('--ADACONTRAST_QUEUE_SIZE', default=None, type=int)
    args = parser.parse_args()
    return args


def merge_cfg_from_args(cfg, args):
    if args.MODEL_CONTINUAL is not None:
        cfg.MODEL.CONTINUAL = args.MODEL_CONTINUAL
    if args.OPTIM_LR is not None:
        cfg.OPTIM.LR = args.OPTIM_LR
    if args.BN_ALPHA is not None:
        cfg.BN.ALPHA = args.BN_ALPHA
    if args.COTTA_RST is not None:
        cfg.COTTA.RST = args.COTTA_RST
    if args.COTTA_AP is not None:
        cfg.COTTA.AP = args.COTTA_AP
    if args.M_TEACHER_MOMENTUM is not None:
        cfg.M_TEACHER.MOMENTUM = args.M_TEACHER_MOMENTUM
    if args.EATA_DM is not None:
        cfg.EATA.D_MARGIN = args.EATA_DM
    if args.EATA_FISHER_ALPHA is not None:
        cfg.EATA.FISHER_ALPHA = args.EATA_FISHER_ALPHA
    if args.T3A_FILTER_K is not None:
        cfg.T3A.FILTER_K = args.T3A_FILTER_K
    if args.LAME_AFFINITY is not None:
        cfg.LAME.AFFINITY = args.LAME_AFFINITY
    if args.LAME_KNN is not None:
        cfg.LAME.KNN = args.LAME_KNN
    if args.TEST_EPOCH is not None:
        cfg.TEST.EPOCH = args.TEST_EPOCH
    if args.SHOT_CLS_PAR is not None:
        cfg.SHOT.CLS_PAR = args.SHOT_CLS_PAR
    if args.SHOT_ENT_PAR is not None:
        cfg.SHOT.ENT_PAR = args.SHOT_ENT_PAR
    if args.NRC_K is not None:
        cfg.NRC.K = args.NRC_K
    if args.NRC_KK is not None:
        cfg.NRC.KK = args.NRC_KK
    if args.SAR_RESET_CONSTANT is not None:
        cfg.SAR.RESET_CONSTANT = args.SAR_RESET_CONSTANT
    if args.PLUE_NUM_NEIGHBORS is not None:
        cfg.PLUE.NUM_NEIGHBORS = args.PLUE_NUM_NEIGHBORS
    if args.ADACONTRAST_NUM_NEIGHBORS is not None:
        cfg.ADACONTRAST.NUM_NEIGHBORS = args.ADACONTRAST_NUM_NEIGHBORS
    if args.ADACONTRAST_QUEUE_SIZE is not None:
        cfg.ADACONTRAST.QUEUE_SIZE = args.ADACONTRAST_QUEUE_SIZE

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os.path as osp
from sklearn.neighbors import KNeighborsClassifier


class FocalLabelSmooth(nn.Module):
    def __init__(self,
                 num_classes,
                 epsilon=0.1,
                 use_gpu=True,
                 size_average=True):
        super(FocalLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.softmax(inputs)

        tmp = log_probs[range(log_probs.shape[0]), targets] # batch
        if self.use_gpu: targets = targets.cuda()
        #targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- (1-tmp)**2 * torch.log(tmp)).mean(0).sum()
        else:
            loss = (-(1 - tmp)**2 * torch.log(tmp)).sum(1)
        return loss


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy



class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self,
                 num_classes,
                 epsilon=0.1,
                 use_gpu=True,
                 size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss


def cal_acc_(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            output_f = netF.forward(inputs)  # a^t
            outputs=netC(output_f)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent




'''def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(), normalize
    ])'''


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])


def image_target(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_shift(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


'''def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(), normalize
    ])'''


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
                       np.array([int(la) for la in val.split()[1:]]))
                      for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1]))
                      for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB'):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def office_load(args):
    train_bs = args.batch_size
    if args.home == True:
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]
        if ss == 'a':
            s = 'Art'
        elif ss == 'c':
            s = 'Clipart'
        elif ss == 'p':
            s = 'Product'
        elif ss == 'r':
            s = 'Real_World'

        if tt == 'a':
            t = 'Art'
        elif tt == 'c':
            t = 'Clipart'
        elif tt == 'p':
            t = 'Product'
        elif tt == 'r':
            t = 'Real_World'

        s_tr, s_ts = './data/office-home/{}.txt'.format(
            s), './data/office-home/{}.txt'.format(s)

        txt_src = open(s_tr).readlines()
        s_tr = txt_src
        s_ts = txt_src

        t_tr, t_ts = './data/office-home/{}.txt'.format(
            t), './data/office-home/{}.txt'.format(t)
        prep_dict = {}
        prep_dict['source'] = image_train()
        prep_dict['target'] = image_target()
        prep_dict['test'] = image_test()
        train_source = ImageList(s_tr,
                                 transform=prep_dict['source'])
        test_source = ImageList(s_ts,
                                transform=prep_dict['source'])
        train_target = ImageList(open(t_tr).readlines(),
                                 transform=prep_dict['target'])
        test_target = ImageList(open(t_ts).readlines(),
                                transform=prep_dict['test'])

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source,
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source,
                                           batch_size=train_bs * 2,#2
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["target"] = DataLoader(train_target,
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["test"] = DataLoader(test_target,
                                      batch_size=train_bs * 3,#3
                                      shuffle=True,
                                      num_workers=args.worker,
                                      drop_last=False)
    return dset_loaders
