# Created on Tue Jan 12 2023 by Chuyang Zhao
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
import torch
from torch import Tensor
from torch import nn


class BaseVideoModel(nn.Module, ABC):
    """
    Base class for all video models.
    """
    def __init__(self, testing=False):
        """
        Args:
            testing (bool): Set to False in testing stage. This is used to
                distinguish model between testing and validation stage,
                because self.training is False in both testing and
                validation stage. In testing mode, we assume target is
                not provided.              

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

    @abstractmethod
    def losses(self, inputs: Tensor, targets: Tensor):
        raise NotImplementedError

    @abstractmethod
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

    def preprocess_inputs(self, batched_inputs: List[Dict]) -> Tensor:
        """
        1. Move the inputs to the current device where model is on.
        2. Collate the list of inputs into a tensor.

        For video dataset, the collated input is expected to have shape of (b, n, c, h, w).

        TODO: if the size of the inputs are not the same, we need to pad them.
        """
        inputs = [self._move_to_current_device(x['input']) for x in batched_inputs]
        inputs = torch.stack(inputs, dim=0)
        return inputs

    def preprocess_targets(self, batched_inputs: List[Dict]) -> Tensor:
        """
        1. Move the targets to the current device where model is on.
        2. Collate the list of targets into a tensor.

        For video dataset, the collated target is expected to have shape of (b, c, h, w).
        """
        targets = [self._move_to_current_device(x['target']) for x in batched_inputs]
        targets = torch.stack(targets, dim=0)
        return targets
        