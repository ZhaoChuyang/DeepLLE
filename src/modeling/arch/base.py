# Created on Tue Oct 11 2022 by Chuyang Zhao
from typing import List, Dict, Optional, Tuple
import torch
from torch import Tensor
from torch import nn

from ..preprocessing import pad_collate_images
from ..postprocessing import remove_padding


class BaseISPModel(nn.Module):
    """
    Base class for all ISP models.
    """
    def __init__(self, testing=False):
        """
        Args:
            testing (bool): Set to False in testing stage. This is used to
                distinguish model between testing and validation stage,
                because self.training is False in both testing and
                validation stage.                

        """
        super().__init__()
        self.testing = testing

        # register a dummy tensor, which will be used to infer the device of the current model
        self.register_buffer("dummy", torch.empty(0), False)
    
    @property
    def device(self):
        return self.dummy.device

    @property
    def size_divisibility(self):
        """
        Some networks require the size of the input image to be divisible
        by some factor, which is often used in encoder-decoder style networks.

        If the network you implemented needs some size divisibility, you can
        override this property, otherwise 1 will be returned which will turn
        off this feature when padding images.
        """
        return 1

    def _move_to_current_device(self, x):
        return x.to(self.device)

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: batched outputs got from data loader.
                For now, each item in the list is a dict contains:
                * image (Tensor): input image in (c, h, w) format.
                * target (Tensor): target image in (c, h, w) format.
                The shapes of the input image and its target should match excatly.

        Returns:
            * in training stage, returns loss_dict and output_dict.
            * in testing stage, returns a list of output images.
        
        """
        raise NotImplementedError

    def preprocess_images(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        1. Pad the batch of input images to the same size and then
        collate them.
        2. Put the input images to the current device.
        """
        images = [self._move_to_current_device(x['image']) for x in batched_inputs]
        images, image_sizes = pad_collate_images(images, self.size_divisibility)
        return images, image_sizes

    def postprocess_images(self, images: Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Post process images returned by the model and then
        remove padding from them.

        Args:
            images (Tensor): processed image tensor (b, c, h, w) returned by the model.
            image_sizes (List[Tuple[int, int]]): list of tuple (h, w) that saves the
                original size of images.

        Returns:
            output_images (List[Tensor]): List of images that is restorted to its original sizes.
        """
        return remove_padding(images, image_sizes)
        