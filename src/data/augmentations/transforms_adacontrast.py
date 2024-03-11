import random
import logging
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
logger = logging.getLogger(__name__)


class NCropsTransform:
    def __init__(self, transform_list) -> None:
        self.transform_list = transform_list

    def __call__(self, x):
        data = [tsfm(x) for tsfm in self.transform_list]
        return data


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_augmentation(aug_type, res_size=256, crop_size=224):
    if aug_type == "moco-v2":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "moco-v2_norm":
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    elif aug_type == "moco-v2-light":
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "moco-v1":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "plain":
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "plain_norm":
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif aug_type == "clip_inference":
        transform_list = [
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    elif aug_type == "test":
        transform_list = [
            transforms.Resize((res_size, res_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    elif aug_type == "imagenet":
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

            # transform = transforms.Compose([
            #     transforms.Resize(224, interpolation=BICUBIC),
            #     transforms.CenterCrop(224),
            #     transforms.ToTensor(),
            #     normalize,
            # ])
        transform_list = [
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        return None

    return transforms.Compose(transform_list)


def get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=256, crop_size=224):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.
    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in aug_versions:
        if version == "s":
            transform_list.append(get_augmentation(aug_type, res_size=res_size, crop_size=crop_size))
        elif version == "w":
            transform_list.append(get_augmentation("plain", res_size=res_size, crop_size=crop_size))
        elif version == "t":
            transform_list.append(get_augmentation("test", res_size=res_size, crop_size=crop_size))
        elif version == "i":
            transform_list.append(get_augmentation("imagenet", res_size=res_size, crop_size=crop_size))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    
    transform = NCropsTransform(transform_list)

    return transform

def get_augmentation_versions_image(aug_versions="twss", aug_type="moco-v2", res_size=256, crop_size=224):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.
    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in aug_versions:
        if version == "s":
            transform_list.append(get_augmentation(aug_type, res_size=res_size, crop_size=crop_size))
        elif version == "w":
            transform_list.append(get_augmentation("plain", res_size=res_size, crop_size=crop_size))
        elif version == "t":
            transform_list.append(get_augmentation("test", res_size=res_size, crop_size=crop_size))
        elif version == "i":
            transform_list.append(get_augmentation("imagenet", res_size=res_size, crop_size=crop_size))
        elif version == "n":
            transform_list.append(get_augmentation("moco-v2_norm", res_size=res_size, crop_size=crop_size))
        elif version == "m":
            transform_list.append(get_augmentation("plain_norm", res_size=res_size, crop_size=crop_size))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    
    transform = NCropsTransform(transform_list)

    return transform

