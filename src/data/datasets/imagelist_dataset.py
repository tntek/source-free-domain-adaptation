# Builds upon: https://github.com/DianCh/AdaContrast/blob/master/image_list.py

import os
import logging
from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional
from torchvision import transforms
import numpy as np

logger = logging.getLogger(__name__)


class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform: Optional[Callable] = None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)

    def build_index(self, label_file):
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            domain = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain))

        return item_list

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label,idx

    def __len__(self):
        return len(self.samples)
    
class ImageList_idx_aug_fix(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform: Optional[Callable] = None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform
        resize_size = 256 
        crop_size = 224 
        #对大于crop_size的图片进行随机裁剪，训练阶段是随机裁剪，验证阶段是随机裁剪或中心裁剪
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#归一化
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                   std=[0.26862954, 0.26130258, 0.27577711])
                                
        # RandomRotate_1 = ts.transforms.RandomRotate(0.5)#以一定的概率（0.5）对图像在[-rotate_range, rotate_range]角度范围内进行旋转
        # self.rf_1 = transforms.Compose([
        #     transforms.Resize((resize_size, resize_size)),
        #     transforms.RandomCrop(crop_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #         normalize
        #     ])
        self.rf_1 = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
                normalize
            ])
        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)

    def build_index(self, label_file):
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            domain = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain))

        return item_list

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_1 = self.transform(img)
            img_2 = self.rf_1(img)
        return img_1,img_2, label,idx

    def __len__(self):
        return len(self.samples)
