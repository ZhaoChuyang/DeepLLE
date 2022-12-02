# Created on Tue Oct 11 2022 by Chuyang Zhao
from typing import List, Dict, Tuple
import torch
from torch import nn, Tensor
from .build import MODEL_REGISTRY
from .base import BaseISPModel
from ..backbone import MBLLEN
from .. import processing
from ..losses import L1Loss, MS_SSIM, SSIM, VGGPerceptualLoss


@MODEL_REGISTRY.register()
class MBLLEN_Arch(BaseISPModel):
    def __init__(self, testing: bool = False):
        super().__init__(testing=testing)
        self.backbone = MBLLEN()
        
        self.l1_loss = L1Loss()
        self.ms_ssim_loss = MS_SSIM(data_range = 1.0)
        self.ssim_loss = SSIM(data_range = 1.0)
        self.perceptual_loss = VGGPerceptualLoss(False)

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        if self.testing:
            images, image_sizes = self.preprocess_test_images(batched_inputs)
            outputs = self.backbone(images)
            # outputs = self.activation(outputs)
            outputs = self.postprocess_images(outputs, image_sizes)
            return outputs
        
        images, targets, image_sizes = self.preprocess_images(batched_inputs)

        # from IPython import embed
        # def save_image(save_path: str, image: torch.Tensor):
        #     """
        #     """
        #     import numpy as np
        #     import cv2 
        #     image = image.detach().cpu().numpy()
        #     image = np.transpose(image, [1, 2, 0])
        #     image = np.clip(image * 255, 0, 255)
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(save_path, image)
        # embed()

        outputs = self.backbone(images)
        # outputs = images * outputs
        # outputs = self.activation(outputs)
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
        loss_dict["perceptual_loss"] = self.perceptual_loss(inputs, targets) * 0.1
        # loss_dict["ms_ssim_loss"] = self.ms_ssim_loss(inputs, targets)
        # loss_dict["ssim_loss"] = (1 - self.ssim_loss(inputs, targets)) * 0.2

        return loss_dict

    def preprocess_images(self, batched_inputs: List[Dict[str, Tensor]]):
        images = [self._move_to_current_device(x['image']) for x in batched_inputs]
        targets = [self._move_to_current_device(x['target']) for x in batched_inputs]

        images, image_sizes = processing.pad_collate_images(images, self.size_divisibility)
        targets, _ = processing.pad_collate_images(targets, self.size_divisibility)

        # images = processing.normalize_to_neg_one_to_one(images)
        # targets = processing.normalize_to_neg_one_to_one(targets)

        return images, targets, image_sizes

    def preprocess_test_images(self, batched_inputs: List[Dict[str, Tensor]]):
        images = [self._move_to_current_device(x['image']) for x in batched_inputs]
        images, image_sizes = processing.pad_collate_images(images, self.size_divisibility)

        # images = processing.normalize_to_neg_one_to_one(images)

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
        # images = processing.unnormalize_to_zero_to_one(images)
        images = processing.remove_padding(images, image_sizes)
        return images

