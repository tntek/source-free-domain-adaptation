import json
import os
import csv
import random
import numpy as np
import scipy.io as sio

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset

# debug dataset
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



class BongardDataset(Dataset):
    def __init__(self, data_root, data_split='unseen_obj_unseen_act', mode='test', 
                        base_transform=None, query_transform=None, with_annotation=False):
        self.base_transform = base_transform
        if query_transform is None:
            self.query_transform = base_transform
        else:
            self.query_transform = query_transform
        self.data_root = data_root
        self.mode = mode
        self.with_annotation = with_annotation
        
        assert mode in ['val', 'test']
        data_file = os.path.join("data/bongard_splits", "bongard_hoi_{}_{}.json".format(self.mode, data_split))
        self.task_list = []
        with open(data_file, "r") as fp:
            task_items = json.load(fp)
            for task in task_items:
                task_data = {}
                pos_samples = []
                neg_samples = []
                for sample in task[0]:
                    neg_samples.append(sample['im_path'])
                for sample in task[1]:
                    pos_samples.append(sample['im_path'])
                
                # random split samples into support and query images (6 vs. 1 for both pos and neg samples) 
                task_data['pos_samples'] = pos_samples
                task_data['neg_samples'] = neg_samples
                task_data['annotation'] = task[-1].replace("++", " ")
                self.task_list.append(task_data)
        
    def __len__(self):
        return len(self.task_list)

    def load_image(self, path, transform_type="base_transform"):
        im_path = os.path.join(self.data_root, path.replace("./", ""))
        if not os.path.isfile(im_path):
            print("file not exist: {}".format(im_path))
            if '/pic/image/val' in im_path:
                im_path = im_path.replace('val', 'train')
            elif '/pic/image/train' in im_path:
                im_path = im_path.replace('train', 'val')
        try:
            image = Image.open(im_path).convert('RGB')
        except:
            print("File error: ", im_path)
            image = Image.open(im_path).convert('RGB')
        trans = getattr(self, transform_type)
        if trans is not None:
            image = trans(image)
        return image

    def __getitem__(self, idx):
        task = self.task_list[idx]
        pos_samples = task['pos_samples']
        neg_samples = task['neg_samples']

        random.seed(0)
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)

        f_pos_support = pos_samples[:-1]
        f_neg_support = neg_samples[:-1]
        pos_images = [self.load_image(f, "base_transform") for f in f_pos_support]
        neg_images = [self.load_image(f, "base_transform") for f in f_neg_support]
        pos_support = torch.stack(pos_images, dim=0)
        neg_support = torch.stack(neg_images, dim=0)

        try:
            pos_query = torch.stack(self.load_image(pos_samples[-1], "query_transform"), dim=0)
            neg_query = torch.stack(self.load_image(neg_samples[-1], "query_transform"), dim=0)
        except:
            pos_query = torch.stack([self.load_image(pos_samples[-1], "query_transform")], dim=0)
            neg_query = torch.stack([self.load_image(neg_samples[-1], "query_transform")], dim=0)

        support_images = torch.cat((pos_support, neg_support), dim=0)
        support_labels = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).long()
        query_images = torch.stack([neg_query, pos_query], dim=0)
        query_labels = torch.Tensor([1, 0]).long()

        if self.with_annotation:
            annotation = task['annotation']
            return support_images, query_images, support_labels, query_labels, annotation
        else:
            return support_images, query_images, support_labels, query_labels




            