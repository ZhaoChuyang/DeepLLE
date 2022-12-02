# Created on Thu Oct 13 2022 by Chuyang Zhao
# modified from: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
import random

import numpy as np
import torch
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


class Compose:
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


class Resize:
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


class RandomResize:
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


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        """
        Args:
            size (int): currently size can only be an int, i.e. only supports rectangle crop.
        """
        self.size = size

    def __call__(self, image, target = None):
        image = pad_if_smaller(image, self.size)
        if target:
            target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if target:
            target = F.crop(target, *crop_params)
        if target:
            return image, target
        return image


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = F.pil_to_tensor(target)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        target = F.convert_image_dtype(target, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        target = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, image, target = None):
        image = F.to_tensor(image)
        if target is None:
            return image
        
        target = F.to_tensor(target)
        return image, target
