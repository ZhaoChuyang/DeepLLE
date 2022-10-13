# Created on Tue Oct 11 2022 by Chuyang Zhao
from typing import List, Tuple, Sequence
from torch import Tensor
import torch


__all__ = ['remove_padding']


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
