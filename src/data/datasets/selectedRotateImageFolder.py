import math
import os
import random

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tr_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                    transforms.ToTensor(),
                                    normalize])
te_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])
te_transforms_imageC = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])

rotation_tr_transforms = tr_transforms
rotation_te_transforms = te_transforms

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


class ImagePathFolder(datasets.ImageFolder):
    def __init__(self, traindir, train_transform):
        super(ImagePathFolder, self).__init__(traindir, train_transform)

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        path, pa = os.path.split(path)
        path, pb = os.path.split(path)
        return img, 'val/%s/%s' % (pb, pa)


# =========================Rotate ImageFolder Preparations Start======================
# Assumes that tensor is (nchannels, height, width)
def tensor_rot_90(x):
    return x.flip(2).transpose(1, 2)


def tensor_rot_180(x):
    return x.flip(2).flip(1)


def tensor_rot_270(x):
    return x.transpose(1, 2).flip(2)


def rotate_single_with_label(img, label):
    if label == 1:
        img = tensor_rot_90(img)
    elif label == 2:
        img = tensor_rot_180(img)
    elif label == 3:
        img = tensor_rot_270(img)
    return img


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        img = rotate_single_with_label(img, label)
        images.append(img.unsqueeze(0))
    return torch.cat(images)


def rotate_batch(batch, label='rand'):
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label
    return rotate_batch_with_labels(batch, labels), labels


# =========================Rotate ImageFolder Preparations End======================


# The following ImageFolder supports sample a subset from the entire dataset by index/classes/sample number, at any time after the dataloader created. 
class SelectedRotateImageFolder(datasets.ImageFolder):
    def __init__(self, root, train_transform, original=True, rotation=True, rotation_transform=None):
        super(SelectedRotateImageFolder, self).__init__(root, train_transform)
        self.original = original
        self.rotation = rotation
        self.rotation_transform = rotation_transform

        self.original_samples = self.samples

    def __getitem__(self, index):
        # path, target = self.imgs[index]
        path, target = self.samples[index]
        img_input = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_input)
        else:
            img = img_input

        results = []
        if self.original:
            results.append(img)
            results.append(target)
        if self.rotation:
            if self.rotation_transform is not None:
                img = self.rotation_transform(img_input)
            target_ssh = np.random.randint(0, 4, 1)[0]
            img_ssh = rotate_single_with_label(img, target_ssh)
            results.append(img_ssh)
            results.append(target_ssh)
        return results

    def switch_mode(self, original, rotation):
        self.original = original
        self.rotation = rotation

    def set_target_class_dataset(self, target_class_index, logger=None):
        self.target_class_index = target_class_index
        self.samples = [(path, idx) for (path, idx) in self.original_samples if idx in self.target_class_index]
        self.targets = [s[1] for s in self.samples]

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = [self.samples[i] for i in indices[:subset_size]]
        self.targets = [self.targets[i] for i in indices[:subset_size]]
        return len(self.targets)

    def set_specific_subset(self, indices):
        self.samples = [self.original_samples[i] for i in indices]
        self.targets = [s[1] for s in self.samples]


def reset_data_sampler(sampler, dset_length, dset):
    sampler.dataset = dset
    if dset_length % sampler.num_replicas != 0 and False:
        sampler.num_samples = math.ceil((dset_length - sampler.num_replicas) / sampler.num_replicas)
    else:
        sampler.num_samples = math.ceil(dset_length / sampler.num_replicas)
    sampler.total_size = sampler.num_samples * sampler.num_replicas


def prepare_train_dataset(args):
    print('Preparing data...')
    traindir = os.path.join(args.data, 'train')
    trset = SelectedRotateImageFolder(traindir, tr_transforms, original=True, rotation=args.rotation,
                                      rotation_transform=rotation_tr_transforms)
    return trset


def prepare_train_dataloader(args, trset=None, sampler=None):
    if sampler is None:
        trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
        train_sampler = None
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trset)
        trloader = torch.utils.data.DataLoader(
            trset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,
            drop_last=True)  # sampler=None shuffle=True,
    return trloader, train_sampler


def prepare_test_data(args, use_transforms=True):
    if args.corruption == 'original':
        te_transforms_local = te_transforms if use_transforms else None
    elif args.corruption in common_corruptions:
        te_transforms_local = te_transforms_imageC if use_transforms else None
    else:
        assert False, NotImplementedError
    if not hasattr(args, 'corruption') or args.corruption == 'original':
        print('Test on the original test set')
        validdir = os.path.join(args.data, 'val')
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                          rotation_transform=rotation_te_transforms)
    elif args.corruption in common_corruptions:
        print('Test on %s level %d' % (args.corruption, args.level))
        validdir = os.path.join(args.data_corruption, args.corruption, str(args.level))
        teset = SelectedRotateImageFolder(validdir, te_transforms_local, original=False, rotation=False,
                                          rotation_transform=rotation_te_transforms)
    else:
        raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 1
    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=args.if_shuffle,
                                           num_workers=args.workers, pin_memory=True)
    return teset, teloader
