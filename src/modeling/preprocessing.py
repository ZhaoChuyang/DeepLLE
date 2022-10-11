# Created on Tue Oct 11 2022 by Chuyang Zhao
from typing import List
import torch
from torch import Tensor


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


# test preprocessing functions.
# checked no fault.
if __name__ == '__main__':
    import os
    import cv2
    import numpy as np

    root = '/data/chuyang/datasets/test/DICM'
    images = []
    filenames = []

    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if not path.endswith(('jpg', 'jpeg', 'png', 'bmp')):
            continue
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, [2, 0, 1])
        img = torch.as_tensor(img)
        images.append(img)
        filenames.append(filename)
    
    batched_images, image_sizes = pad_collate_images(images, 32)

    processed_images = []

    for image in batched_images:
        image = image.numpy()
        image = np.transpose(image, [1, 2, 0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed_images.append(image)

    os.mkdir(root + '/test')
    for filename, image in zip(filenames, processed_images):
        print(root + '/test/' + filename)
        cv2.imwrite(root + '/test/' + filename, image)
