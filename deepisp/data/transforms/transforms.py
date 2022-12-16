# Created on Thu Oct 13 2022 by Chuyang Zhao
# modified from: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
import random
import numpy as np
import torch
from abc import ABC, abstractmethod
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img



class Transform(ABC):
    def __init__(self):
        """
        Base class for all transforms. For ISP tasks, we often
        need to process input with a pair of images, i.e. input
        image and target image. So the transform function need to
        take these two as inputs and do the same transforms on them.
        
        In inference stage, the target image may not provided, so
        the transforms should function correctly with only one input.
        """
        pass

    @abstractmethod
    def __call__(self, image, target = None):
        pass


class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target = None):
        if target is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target


class Resize(Transform):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, image, target = None):
        size = self.size
        image = F.resize(image, size)
        if target is None:
            return image
        target = F.resize(target, size)
        return image, target


class RandomResize(Transform):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size)
        return image, target


class RandomHorizontalFlip(Transform):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target



class RandomVerticalFlip(Transform):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target


class RandomCrop(Transform):
    def __init__(self, size):
        """
        Args:
            size (int): currently size can only be an int, i.e. only supports rectangle crop.
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, image, target = None):
        image = pad_if_smaller(image, min(self.size))
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if not target:
            return image

        target = pad_if_smaller(target, min(self.size))
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(Transform):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target = None):
        image = F.center_crop(image, self.size)
        if not target: 
            return image
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor(Transform):
    def __call__(self, image, target = None):
        image = F.pil_to_tensor(image)
        if not target:
            return image
        target = F.pil_to_tensor(target)
        return image, target


class ConvertImageDtype(Transform):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target = None):
        image = F.convert_image_dtype(image, self.dtype)
        if not target:
            return image
        target = F.convert_image_dtype(target, self.dtype)
        return image, target


class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target = None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if not target:
            return image
        target = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor(Transform):
    def __init__(self):
        pass

    def __call__(self, image, target = None):
        image = F.to_tensor(image)
        if target is None:
            return image
        target = F.to_tensor(target)
        return image, target


class RandomRightRotation(Transform):
    def __init__(self, p: float):
        self.p = p
    
    def __call__(self, image: torch.Tensor, target: torch.Tensor = None):
        # don't do augmentation in inference mode
        if target is None:
            return image
        
        if random.random() < self.p:
            rot_code = random.randint(1, 3)
            degree = rot_code * 90
            image = F.rotate(image, degree)
            target = F.rotate(target, degree)
        
        return image, target


class IdentityAug(Transform):
    def __init__(self, p):
        """
        Construct an identity pair of target images given the probability p.
        This is observed to improve the perceptual quality of the results.
        """
        self.p = p

    def __call__(self, image, target = None):
        if target is None:
            return image
        
        if random.random() < self.p:
            image = target
        
        return image, target
