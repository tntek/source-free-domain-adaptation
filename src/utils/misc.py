import os
import math
import torch
from colorama import Fore

def get_grad(params):
	if isinstance(params, torch.Tensor):
		params = [params]
	params = list(filter(lambda p: p.grad is not None, params))
	grad = [p.grad.data.cpu().view(-1) for p in params]
	return torch.cat(grad)

def write_to_txt(name, content):
	with open(name, 'w') as text_file:
		text_file.write(content)

def my_makedir(name):
	try:
		os.makedirs(name)
	except OSError:
		pass

def print_args(opt):
	for arg in vars(opt):
		print('%s %s' % (arg, getattr(opt, arg)))

def mean(ls):
	return sum(ls) / len(ls)

def normalize(v):
	return (v - v.mean()) / v.std()

def flat_grad(grad_tuple):
    return torch.cat([p.view(-1) for p in grad_tuple])

def print_nparams(model):
	nparams = sum([param.nelement() for param in model.parameters()])
	print('number of parameters: %d' % (nparams))

def print_color(color, string):
	print(getattr(Fore, color) + string + Fore.RESET)

def freeze_params(model):
    for name, p in model.named_parameters():
        p.requires_grad = False
    print("Freeze parameter until", name)

def print_params(model):
    for name, p in model.named_parameters():
        print(name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr

    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (
            1 + math.cos(math.pi * epoch / args.nepoch)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
