from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data

class DomainNet(data.Dataset):
    def __init__(self, root, domain, train=True, transform=None, from_file=False):
        
        if not from_file:
            data = []
            labels = []

            f = open(os.path.join(root,domain+"_list.txt"), "r")
            lines = f.readlines()
            lines = [l.split(" ") for l in lines]
            lines = np.array(lines)

            files = lines[:-1,0]
            files = [os.path.join(root, sfile) for sfile in files]

            classes = lines[:-1,1]
            classes = [int(c[:-1]) for c in classes]

            data.extend(files)
            labels.extend(classes)

            self.data = np.array(data)
            self.labels = np.array(labels)

        else:
            data = np.load(os.path.join(root, domain+"_imgs.npy"))
            labels = np.load(os.path.join(root, domain+"_labels.npy"))
        
            np.random.seed(1234)
            idx = np.random.permutation(len(data))

            self.data = np.array(data)[idx]
            self.labels = np.array(labels)[idx]

            test_perc = 20
            
            test_len = len(self.data)*test_perc//100                   
            
            if train:
                self.data = self.data[test_len:]
                self.labels = self.labels[test_len:]
            else:
                self.data = self.data[:test_len]
                self.labels = self.labels[:test_len]

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
        return len(self.X)

