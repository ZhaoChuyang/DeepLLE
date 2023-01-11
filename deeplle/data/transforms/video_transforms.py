from abc import ABC, abstractmethod
import random
from typing import List, Union, Tuple
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T


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
        Base class for all transforms for video data. Video data
        should have the following format:
            * input (list): a list of images of size n
            * target (optional): single image or None
        n should be an odd number and the (n//2+1)-th image is the input image paired with target.

        For training dataset, both input and target should be provided. But
        for inference/testing dataset, only input is required. So for each
        transform method inherited from Transform, it should be able to handle
        these both these two situations.
        """
        pass

    @abstractmethod
    def __call__(self, input: List, target = None):
        """
        Args:
            input (list): list of images.
            target (optional): target image or None.

        Returns:
            input: transformed input images, should be a list of images or a tensor.
            target: transformed target image or None. 
        """
        pass


class Compose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target = None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize(Transform):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, input: List, target = None):
        size = self.size
        for i in range(len(input)):
            input[i] = F.resize(input[i], size)
        target = F.resize(target, size) if target is not None else None
        return input, target


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        """
        Flip the input and target horizontally with a probability of p.

        Args:
            p (float): probability of flipping the image. Default value is 0.5.
        """
        self.p = p

    def __call__(self, input: List, target = None):
        if random.random() < self.p:
            for i in range(len(input)):
                input[i] = F.hflip(input[i])
            target = F.hflip(target) if target is not None else None
        return input, target


class RandomVerticalFlip(Transform):
    def __init__(self, p: float = 0.5):
        """
        Flip the input and target horizontally with a probability of p.

        Args:
            p (float): probability of flipping the image. Default value is 0.5.
        """
        self.p = p

    def __call__(self, input: List, target = None):
        if random.random() < self.p:
            for i in range(len(input)):
                input[i] = F.vflip(input[i])
            target = F.vflip(target) if target is not None else None
        return input, target


class RandomCrop(Transform):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Args:
            size (int): currently size can only be an int, i.e. only supports rectangle crop.
        """
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, input: List, target = None):
        for i in range(len(input)):
            input[i] = pad_if_smaller(input[i], min(self.size))
            crop_params = T.RandomCrop.get_params(input[i], self.size)
            input[i] = F.crop(input[i], *crop_params)
            
        if target is not None:
            target = pad_if_smaller(target, min(self.size))
            target = F.crop(target, *crop_params)

        return input, target


class ToTensor(Transform):
    def __init__(self):
        """
        Transform input and target to tensor.
        
        It takes two arguments input and target, input is a list of n images,
        target is a single image or None.

        The transformed input is a tensor of shape (n, c, h, w)
        and target is a tensor of shape (c, h, w) is target is not None.

        NOTE: This should be the last transform operation, since its return values 
        are two tensors and other transforms returned a list and an image.
        """
        pass

    def __call__(self, input: List, target = None):
        for i in range(len(input)):
            input[i] = F.to_tensor(input[i])
        input = torch.stack(input)
        target = F.to_tensor(target) if target is not None else None
        return input, target
