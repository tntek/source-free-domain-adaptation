import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

def prepare_transforms(dataset):

    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise NotImplementedError

    normalize = transforms.Normalize(mean=mean, std=std)

    te_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    tr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    simclr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),     # TODO: modify the hard-coded size
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])

    return tr_transforms, te_transforms, simclr_transforms

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

# -------------------------

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_mix_corruption(args, num_mix, foldername):
    tesize = 10000
    if num_mix == 10:
        ## Contrast
        teset_c_raw = np.load(foldername + '/contrast.npy')
        teset_c_raw = teset_c_raw[(args.level-1)*tesize: args.level*tesize][:1000]

        ## De_focus
        teset_d_raw = np.load(foldername + '/defocus_blur.npy')
        teset_d_raw = teset_d_raw[(args.level-1)*tesize: args.level*tesize][1000:2000]

        ## Elastic
        teset_e_raw = np.load(foldername + '/elastic_transform.npy')
        teset_e_raw = teset_e_raw[(args.level-1)*tesize: args.level*tesize][2000:3000]

        ## Fog
        teset_f_raw = np.load(foldername + '/fog.npy')
        teset_f_raw = teset_f_raw[(args.level-1)*tesize: args.level*tesize][3000:4000]

        ## Frost
        teset_fr_raw = np.load(foldername + '/frost.npy')
        teset_fr_raw = teset_fr_raw[(args.level-1)*tesize: args.level*tesize][4000:5000]

        ## Gaussian
        teset_g_raw = np.load(foldername + '/gaussian_noise.npy')
        teset_g_raw = teset_g_raw[(args.level-1)*tesize: args.level*tesize][5000:6000]

        ## Glass
        teset_gl_raw = np.load(foldername + '/glass_blur.npy')
        teset_gl_raw = teset_gl_raw[(args.level-1)*tesize: args.level*tesize][6000:7000]

        ## Impulse
        teset_i_raw = np.load(foldername + '/impulse_noise.npy')
        teset_i_raw = teset_i_raw[(args.level-1)*tesize: args.level*tesize][7000:8000]

        ## JPEG
        teset_j_raw = np.load(foldername + '/jpeg_compression.npy')
        teset_j_raw = teset_j_raw[(args.level-1)*tesize: args.level*tesize][8000:9000]

        ## Motion
        teset_m_raw = np.load(foldername + '/motion_blur.npy')
        teset_m_raw = teset_m_raw[(args.level-1)*tesize: args.level*tesize][9000:]

        teset_mix_raw = np.concatenate([teset_c_raw, teset_d_raw, teset_e_raw, teset_f_raw, teset_fr_raw, teset_g_raw, teset_gl_raw, teset_i_raw, teset_j_raw, teset_m_raw])

    elif num_mix == 5:
        ## Frost
        teset_fr_raw = np.load(foldername + '/frost.npy')
        teset_fr_raw = teset_fr_raw[(args.level-1)*tesize: args.level*tesize][:2000]

        ## Gaussian
        teset_g_raw = np.load(foldername + '/gaussian_noise.npy')
        teset_g_raw = teset_g_raw[(args.level-1)*tesize: args.level*tesize][2000:4000]

        ## Glass
        teset_gl_raw = np.load(foldername + '/glass_blur.npy')
        teset_gl_raw = teset_gl_raw[(args.level-1)*tesize: args.level*tesize][4000:6000]

        ## Impulse
        teset_i_raw = np.load(foldername + '/impulse_noise.npy')
        teset_i_raw = teset_i_raw[(args.level-1)*tesize: args.level*tesize][6000:8000]

        ## JPEG
        teset_j_raw = np.load(foldername + '/jpeg_compression.npy')
        teset_j_raw = teset_j_raw[(args.level-1)*tesize: args.level*tesize][8000:]

        teset_mix_raw = np.concatenate([teset_fr_raw, teset_g_raw, teset_gl_raw, teset_i_raw, teset_j_raw])
    else:
        raise NotImplementedError

    return teset_mix_raw

def prepare_test_data(args, ttt=False, num_sample=None):

    tr_transforms, te_transforms, simclr_transforms = prepare_transforms(args.dataset)

    if args.dataset == 'cifar10':

        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            print('Test on the original test set')
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=te_transforms)
        elif args.corruption in common_corruptions:
            print('Test on %s level %d' %(args.corruption, args.level))
            teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
            teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=te_transforms)
            teset.data = teset_raw

        elif args.corruption == 'cifar_new':
            from utils.cifar_new import CIFAR_New
            print('Test on CIFAR-10.1')
            teset = CIFAR_New(root=args.dataroot + '/CIFAR-10.1/datasets/', transform=te_transforms)
            permute = False

        elif args.corruption == 'cifar_mix10':
            print('Test on mix on 10 noises on level %d' %(args.level))
            teset_mix_raw = prepare_mix_corruption(args, 10, args.dataroot + '/CIFAR-10-C')
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=te_transforms)
            teset.data = teset_mix_raw

        elif args.corruption == 'cifar_mix5':
            print('Test on mix on 5 noises on level %d' %(args.level))
            teset_mix_raw = prepare_mix_corruption(args, 5, args.dataroot + '/CIFAR-10-C')
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                                train=False, download=True, transform=te_transforms)
            teset.data = teset_mix_raw

        else:
            raise Exception('Corruption not found!')

    elif args.dataset == 'cifar100':
     
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            print('Test on the original test set')
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                                train=False, download=True, transform=te_transforms)
        elif args.corruption in common_corruptions:
            print('Test on %s level %d' %(args.corruption, args.level))
            teset_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
            teset_raw = teset_raw[(args.level-1)*tesize: args.level*tesize]
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                                train=False, download=True, transform=te_transforms)
            teset.data = teset_raw
        else:
            raise Exception('Corruption not found!')

    else:
        raise Exception('Dataset not found!')

    if not hasattr(args, 'workers') or args.workers < 2:
        pin_memory = False
    else:
        pin_memory = True

    if ttt:
        shuffle = True
        drop_last = True
    else:
        shuffle = True
        drop_last = False

    if num_sample and num_sample < teset.data.shape[0]:
        teset.data = teset.data[:num_sample]
        print("Truncate the test set to {:d} samples".format(num_sample))

    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                            shuffle=shuffle, num_workers=args.workers,
                                            worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=drop_last)
    return teset, teloader

def prepare_train_data(args, num_sample=None):
    print('Preparing data...')
    
    tr_transforms, te_transforms, simclr_transforms = prepare_transforms(args.dataset)

    if args.dataset == 'cifar10':

        if hasattr(args, 'ssl') and args.ssl == 'contrastive':
            trset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                         train=True, download=True,
                                         transform=TwoCropTransform(simclr_transforms))
            if hasattr(args, 'corruption') and args.corruption in common_corruptions:
                print('Contrastive on %s level %d' %(args.corruption, args.level))
                tesize = 10000
                trset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
                trset_raw = trset_raw[(args.level-1)*tesize: args.level*tesize]   
                trset.data = trset_raw
            elif hasattr(args, 'corruption') and args.corruption == 'cifar_new':
                from utils.cifar_new import CIFAR_New
                print('Contrastive on CIFAR-10.1')
                trset_raw = CIFAR_New(root=args.dataroot + '/CIFAR-10.1/datasets/', transform=te_transforms)
                trset.data = trset_raw.data
            elif  hasattr(args, 'corruption') and args.corruption == 'cifar_mix10':
                print('Test on mix on 10 noises on level %d' %(args.level))
                trset_mix_raw = prepare_mix_corruption(args, 10, args.dataroot + '/CIFAR-10-C')
                trset.data = trset_mix_raw
            elif  hasattr(args, 'corruption') and args.corruption == 'cifar_mix5':
                print('Test on mix on 5 noises on level %d' %(args.level))
                trset_mix_raw = prepare_mix_corruption(args, 5, args.dataroot + '/CIFAR-10-C')
                trset.data = trset_mix_raw
            else:
                print('Contrastive on ciar10 training set')
        else:
            trset = torchvision.datasets.CIFAR10(root=args.dataroot,
                                        train=True, download=True, transform=tr_transforms)
            print('Cifar10 training set')

    elif args.dataset == 'cifar100':
        if hasattr(args, 'ssl') and args.ssl == 'contrastive':
            trset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                         train=True, download=True,
                                         transform=TwoCropTransform(simclr_transforms))            
            if hasattr(args, 'corruption') and args.corruption in common_corruptions:
                print('Contrastive on %s level %d' %(args.corruption, args.level))
                tesize = 10000
                trset_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
                trset_raw = trset_raw[(args.level-1)*tesize: args.level*tesize]   
                trset.data = trset_raw
            else:
                print('Contrastive on ciar10 training set')
        else:
            trset = torchvision.datasets.CIFAR100(root=args.dataroot,
                                            train=True, download=True, transform=tr_transforms)
            print('Cifar100 training set')
    else:
        raise Exception('Dataset not found!')

    if not hasattr(args, 'workers') or args.workers < 2:
        pin_memory = False
    else:
        pin_memory = True

    if num_sample and num_sample < trset.data.shape[0]:
        trset.data = trset.data[:num_sample]
        print("Truncate the training set to {:d} samples".format(num_sample))

    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers,
                                            worker_init_fn=seed_worker, pin_memory=pin_memory, drop_last=True)
    return trset, trloader
