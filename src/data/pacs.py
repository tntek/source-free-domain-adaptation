from __future__ import print_function
from PIL import Image
import os
import os.path
from os.path import join
import numpy as np
import sys
import torchvision.transforms as transforms
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch.utils.data as data
import glob


class PACS(data.Dataset):
    def __init__(self, root, domain, train=True, transform=None, from_file=False):
        self.train = train
        self.transform = transform
        
        if not from_file:
            data = []
            labels = []
            for c in range(7):
                files = glob.glob(os.path.join(root,domain,str(c),"*"))
                data.extend(files)
                labels.extend([c]*len(files))
        
            np.random.seed(1234)
            idx = np.random.permutation(len(data))

            self.data = np.array(data)[idx]
            self.labels = np.array(labels)[idx]

            test_perc = 20
            
            test_len = len(self.data)*test_perc//100                   
            
            if self.train:
                self.data = self.data[test_len:]
                self.labels = self.labels[test_len:]
            else:
                self.data = self.data[:test_len]
                self.labels = self.labels[:test_len]
        else:
            mode = 'train' if self.train else 'test'
            self.data = np.load(os.path.join(root, domain+"_"+mode+"_imgs.npy"))
            self.labels = np.load(os.path.join(root, domain+"_"+mode+"_labels.npy"))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.labels[index]          

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)
         
        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.data)

