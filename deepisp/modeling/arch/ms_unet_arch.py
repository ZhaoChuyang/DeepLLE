# Created on Tue Oct 11 2022 by Chuyang Zhao
from typing import List, Dict, Tuple
import torch
from torch import nn, Tensor
from ..build import MODEL_REGISTRY
from .base import BaseISPModel
from ..backbone import UNet
from .. import processing
from ..losses import L1Loss, MS_SSIM, SSIM, VGGPerceptualLoss


"""
For debugging.
"""
def save_image(save_path: str, image: torch.Tensor):
    import numpy as np
    import cv2 
    image = image.detach().cpu().numpy()
    image = np.transpose(image, [1, 2, 0])
    image = np.clip(image * 255, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)


@MODEL_REGISTRY.register()
class MultiStageUNet(BaseISPModel):
    def __init__(self, bilinear: bool = False, depth: int = 2, base_dim: int = 32, residual: bool = False, testing: bool = False):
        """
        Multi-stage architecture is widely used in image restoration, some of the SOTA methods in image restoration,
        deblurring resort to this architecture, i.e., HINet, MPRNet, etc.
        """
        super(MultiStageUNet, self).__init__(testing=testing)
        self.register_buffer("counter", tensor=torch.zeros(size=(1,)))

        self.stage_1 = UNet(n_channels=3, n_classes=3, bilinear=bilinear, base_dim=base_dim, depth=depth, residual=residual)
        self.stage_2 = UNet(n_channels=6, n_classes=3, bilinear=bilinear, base_dim=base_dim, depth=depth, residual=residual)
        
        self.l1_loss = L1Loss()
        self.ms_ssim_loss = MS_SSIM(data_range = 1.0)
        self.ssim_loss = SSIM(data_range = 1.0)
        self.perceptual_loss = VGGPerceptualLoss(False)
        


    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        if self.testing:
            images, image_sizes = self.preprocess_test_images(batched_inputs)
            x = self.stage_1(images)
            x = torch.cat([x, images], dim=1)
            x = self.stage_2(x)
            # outputs = self.activation(outputs)
            outputs = self.postprocess_images(x, image_sizes)
            return outputs
        
        images, targets, image_sizes = self.preprocess_images(batched_inputs)



        x1 = self.stage_1(images)
        x = torch.cat([x1, images], dim=1)
        x2 = self.stage_2(x) 

        loss_dict_1 = self.losses(x1, targets, 1)
        loss_dict_2 = self.losses(x2, targets, 2)
        if self.counter % 2 == 0:
            loss_dict = loss_dict_1
        else:
            loss_dict = loss_dict_2
        self.counter += 1
        # loss_dict.update(loss_dict_2)

        output_dict = self.metrics(x2, targets)

        return loss_dict, output_dict

    @torch.no_grad()
    def metrics(self, inputs: Tensor, targets: Tensor):
        output_dict = {}
        return output_dict

    def losses(self, inputs: Tensor, targets: Tensor, stage: int):
        assert inputs.shape == targets.shape, "Shapes of inputs: {} and targets: {} do not match.".format(inputs.shape, targets.shape)
        loss_dict = {}

        loss_dict[f"stage_{stage}_l1_loss"] = self.l1_loss(inputs, targets)
        loss_dict[f"stage_{stage}_perceptual_loss"] = self.perceptual_loss(inputs, targets) * 0.1
        loss_dict[f"stage_{stage}_ms_ssim_loss"] = (1 - self.ms_ssim_loss(inputs, targets))
        # loss_dict["ssim_loss"] = (1 - self.ssim_loss(inputs, targets))

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

