# Created on Tue Oct 11 2022 by Chuyang Zhao
from typing import List, Tuple
from torch import Tensor


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
