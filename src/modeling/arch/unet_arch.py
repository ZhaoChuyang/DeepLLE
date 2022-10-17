# Created on Tue Oct 11 2022 by Chuyang Zhao
from typing import List, Dict
import torch
from torch import nn, Tensor
from .build import MODEL_REGISTRY
from .base import BaseISPModel
from ..backbone import UNet
from .. import preprocessing
from ..losses import L1Loss, MS_SSIM


@MODEL_REGISTRY.register()
class UNetBaseline(BaseISPModel):
    def __init__(self, bilinear: bool = False, testing: bool = False):
        super().__init__(testing=testing)
        self.backbone = UNet(n_channels=3, n_classes=3, bilinear=bilinear)
        
        self.l1_loss = L1Loss()
        self.ms_ssim_loss = MS_SSIM(data_range=1)

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        if self.testing:
            images, image_sizes = self.preprocess_test_images(batched_inputs)
            outputs = self.backbone(images)
            outputs = self.postprocess_images(outputs, image_sizes)
            return outputs
        
        images, targets, image_sizes = self.preprocess_images(batched_inputs)

        outputs = self.backbone(images)
        loss_dict = self.losses(outputs, targets)
        output_dict = self.metrics(outputs, targets)

        return loss_dict, output_dict

    @torch.no_grad()
    def metrics(self, inputs: Tensor, targets: Tensor):
        output_dict = {}
        return output_dict

    def losses(self, inputs: Tensor, targets: Tensor):
        assert inputs.shape == targets.shape, "Shapes of inputs: {} and targets: {} do not match.".format(inputs.shape, targets.shape)
        loss_dict = {}

        loss_dict["l1_loss"] = self.l1_loss(inputs, targets)
        loss_dict["ms_ssim_loss"] = self.ms_ssim_loss(inputs, targets)

        return loss_dict

    def preprocess_images(self, batched_inputs: List[Dict[str, Tensor]]):
        images = [self._move_to_current_device(x['image']) for x in batched_inputs]
        targets = [self._move_to_current_device(x['target']) for x in batched_inputs]

        images, image_sizes = preprocessing.pad_collate_images(images, self.size_divisibility)
        targets, _ = preprocessing.pad_collate_images(targets, self.size_divisibility)

        return images, targets, image_sizes

    def preprocess_test_images(self, batched_inputs: List[Dict[str, Tensor]]):
        images = [self._move_to_current_device(x['image']) for x in batched_inputs]
        images, image_sizes = preprocessing.pad_collate_images(images, self.size_divisibility)
        return images, image_sizes
