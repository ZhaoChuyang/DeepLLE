# Created on Tue Oct 11 2022 by Chuyang Zhao
from typing import List, Sequence, Tuple
import torch
from torch import Tensor
import random


__all__ = ['pad_collate_images']


def pad_collate_images(images: List[Tensor], size_divisibility: int = 1, pad_value: float = 0.0):
    """
    Pad the input images of shape to the same size
    and then collate them into a Tensor of shape.

    Padded size should satisify two rules:
    1. padded size >= the max size of all images
    2. padded size should be divisible by size_divisibility

    Args:
        images (List[Tensor]): List of images of different size.
        size_divisibility (int): If size_divisibility > 1, add padding
            to ensure the padded height and width is divisible by 
            size_divisibility. This depends on the model and many models
            need a divisibility of 32.
        pad_value (float): Value to pad.
    
    Returns:
        batched_images (Tensor): Batched images of shape (B, C, H, W),
            B is the size of input images, i.e. B = len(images).
        image_sizes (List[Tuple[int, int]]): List of the original input
            image sizes. Each tuple is (H, W).
    """
    assert len(images) > 0
    assert isinstance(images, (tuple, list))
    for t in images:
        assert isinstance(t, Tensor), type(t)
        assert t.shape[:-2] == images[0].shape[:-2], t.shape
    
    image_sizes = [(im.shape[-2], im.shape[-1]) for im in images]
    image_sizes_tensor = torch.as_tensor(image_sizes)

    # max_size: (2,) max values of h and w
    max_size = image_sizes_tensor.max(0).values

    if size_divisibility > 1:
        stride = size_divisibility
        max_size = torch.floor((max_size + (stride - 1)).div(stride)) * stride
    
    batch_shape = [len(images)] + list(images[0].shape[:-2]) + [int(max_size[0]), int(max_size[1])]

    batched_images = images[0].new_full(batch_shape, pad_value)
    batched_images = batched_images.to(images[0].device)
    
    for i, img in enumerate(images):
        batched_images[i, ..., :img.shape[-2], :img.shape[-1]].copy_(img)
    
    return batched_images.contiguous(), image_sizes


def normalize_to_neg_one_to_one(images: Tensor):
    """
    Normalize values of tensor from [0, 1] to [-1, 1].

    Args:
        images (tensor): input tensor.
    
    Returns:
        tensor normalized to [-1, 1].
    """
    return images * 2 - 1


def unnormalize_to_zero_to_one(images: Tensor):
    """
    Unnormalize the values of a tensor from [-1, 1] to [0, 1]
    
    Args:
        images (tensor): input tensor.

    Returns:
        tensor normalized to [0, 1].
    """
    return (images + 1) * 0.5


def remove_padding(images: Tensor, image_sizes: List[Tuple[int, int]]):
    """
    Remove paddings from images and returns copy of them of the size in image_sizes.

    Args:
        images (Tensor): input image tensor of shape (b, c, h, w)
        image_sizes (List[Tuple[int, int]]): list of the original sizes of the images.
            tuple is of (h, w).
    
    Returns:
        output_images (List[Tensor]): return the processed image tensor in a list,
            because the shape of the processed images are not same, so they can not
            be stacked in a tensor, we need to save them in a list.
    """
    output_images = []

    for i, image_size in enumerate(image_sizes):
        output_images.append(images[i, ..., :image_size[0], :image_size[1]])
    
    return output_images


def unnormalize(images: Tensor, mean: Sequence, std: Sequence):
    """
    unnormalize the z-score normalized images to the original data range.

    Args:
        images (Tensor): images of (b, c, h, w)
        mean (Sequence): original z-score mean applied to images, len(mean) == c.
        std (Sequence): original z-score std applied to images, len(std) == c.

    Returns:
        images reversed to its original data range, commonly from 0 to 1.
    """
    b, c, h, w = images.shape

    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)

    assert len(mean) == len(std) == c, "channel number of mean: {}, std: {}, and images: {} should match.".format(len(mean), len(std), images.shape)

    mean = mean.view(1, -1, 1, 1)
    std = mean.view(1, -1, 1, 1)

    mean = mean.repeat(b, 1, h, w)
    std = std.repeat(b, 1, h, w)

    images = (images * std) + mean

    return images


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_



# test preprocessing functions.
# checked no fault.
if __name__ == '__main__':
    import os
    import cv2
    import numpy as np

    root = '/data/chuyang/datasets/test/DICM'
    images = []
    filenames = []

    """
    Test pad_collate_images
    """
    # for filename in os.listdir(root):
    #     path = os.path.join(root, filename)
    #     if not path.endswith(('jpg', 'jpeg', 'png', 'bmp')):
    #         continue
    #     img = cv2.imread(path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = np.transpose(img, [2, 0, 1])
    #     img = torch.as_tensor(img)
    #     images.append(img)
    #     filenames.append(filename)
    
    # batched_images, image_sizes = pad_collate_images(images, 32)

    # processed_images = []

    # for image in batched_images:
    #     image = image.numpy()
    #     image = np.transpose(image, [1, 2, 0])
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     processed_images.append(image)

    # os.mkdir(root + '/test')
    # for filename, image in zip(filenames, processed_images):
    #     print(root + '/test/' + filename)
    #     cv2.imwrite(root + '/test/' + filename, image)

    """
    Test Mixing Augment
    """
    images = []
    aug = Mixing_Augment(mixup_beta=1.2, use_identity=True, device=torch.device('cpu'))
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if not path.endswith(('jpg', 'jpeg', 'png', 'bmp')):
            continue
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, [2, 0, 1])
        img = torch.as_tensor(img)
        img, _ = aug(img, img)
        images.append(img)
        filenames.append(filename)
    
    processed_images = []
    
    for image in images:
        image = image.numpy()
        image = np.transpose(image, [1, 2, 0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed_images.append(image)

    os.mkdir(root + '/test')
    for filename, image in zip(filenames, processed_images):
        print(root + '/test/' + filename)
        cv2.imwrite(root + '/test/' + filename, image)
