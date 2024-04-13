from copy import deepcopy
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from src.utils import IID_losses
from clip.custom_clip import get_coop
from data.datautils_domain import  build_dataset
from data.cls_to_names import *
from data.domain_datasets import domain_datasets


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

def image_test_50(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=[0.26862954, 0.26130258, 0.27577711])
                            
    return transforms.Compose([
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])

def test_time_tuning(model, inputs,pesu_label, optimizer, cfg):
    for j in range(cfg.DIFO.TTA_STEPS):
        with torch.cuda.amp.autocast():
            output,_ = model(inputs) 
            pesu_label = pesu_label.cuda()
            output = nn.Softmax(dim=1)(output)
            loss = IID_losses.IID_loss(output, pesu_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 

def prompt_main(cfg,confi_imag,confi_dis,iter_num):
    # This codebase has only been tested under the single GPU setting
    assert int(cfg.GPU_ID) is not None
    text_features = main_worker(cfg,confi_imag,confi_dis)
    text_features = text_features.detach()
    return text_features


def main_worker(cfg,confi_imag,confi_dis):
    if cfg.SETTING.DATASET in domain_datasets:
        cfg.domain_name = cfg.domain[cfg.SETTING.T]
        classnames = cfg.classname

    model = get_coop(cfg.DIFO.ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.DIFO.N_CTX, cfg.DIFO.CTX_INIT)
    model = model.cuda()

    if cfg.DIFO.LOAD is not None:
        print("loading prompt")
        pretrained_ctx = torch.load(cfg.DIFO.LOAD)['ctx']
        assert pretrained_ctx.size()[0] == cfg.DIFO.N_CTX
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(pretrained_ctx)
            model.prompt_learner.ctx_init_state = pretrained_ctx

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    trainable_param = model.prompt_learner.parameters()
    if 'RN' in cfg.DIFO.ARCH :
        prompt_lr = cfg.OPTIM.LR*0.1
        data_transform = image_test_50()
    else :
        data_transform = image_test()
        prompt_lr = cfg.OPTIM.LR

    optimizer = torch.optim.SGD(trainable_param, prompt_lr,weight_decay=5e-4,momentum=0.9,nesterov=False)
    optim_state = deepcopy(optimizer.state_dict())
    cudnn.benchmark = True
    set_id = 'sfuda'
    model.reset_classnames(classnames, cfg.DIFO.ARCH)

    val_dataset = build_dataset(set_id, data_transform,confi_imag,confi_dis,cfg.DATA_DIR,cfg.domain_name,mode='test')
    batchsize = cfg.TEST.BATCH_SIZE
    val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batchsize, shuffle=True,
                num_workers=cfg.NUM_WORKERS,drop_last=False)    
    text_features = test_time_adapt_eval(val_loader, model, optimizer, optim_state, cfg)
    return text_features

def test_time_adapt_eval(val_loader, model, optimizer, optim_state, cfg):
    with torch.no_grad():
        model.train()
    max_iter = len(val_loader)
    iter_num = 0
    while iter_num < max_iter:
        try:
            images, target,pesu_label,_ = next(iter_test)
        except:
            iter_test = iter(val_loader)
            images, target,pesu_label,_ = next(iter_test)

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)
        images = images.cuda(int(cfg.GPU_ID), non_blocking=True)
        image = images
        target = target.cuda(int(cfg.GPU_ID), non_blocking=True)
        
        if cfg.DIFO.TTA_STEPS > 0:
            with torch.no_grad():
                model.train()
        optimizer.load_state_dict(optim_state)
        test_time_tuning(model,images,pesu_label, optimizer, cfg)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                model.eval()
                _,text_features = model(image)
        iter_num = iter_num + 1
    torch.save(model.prompt_learner.state_dict(),"prompt_model.pt")
    return text_features