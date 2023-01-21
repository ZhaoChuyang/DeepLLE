from typing import List, Dict, Tuple
import torch
from torch import nn, Tensor
from deeplle.modeling.arch.base_image_model import BaseISPModel
from deeplle.modeling.build import MODEL_REGISTRY
from deeplle.modeling.backbone.iat import IAT
from deeplle.modeling.losses import L1Loss, MS_SSIM, SSIM, VGGPerceptualLoss, CharbonnierLoss, TVLoss
from deeplle.modeling import processing


__all__ = ['IlluminationAdaptiveTransformer']


@MODEL_REGISTRY.register()
class IlluminationAdaptiveTransformer(BaseISPModel):
    def __init__(self, in_dim: int = 3, with_global: bool = True, testing: bool = False):
        super().__init__(testing=testing)
        self.backbone = IAT(in_dim = in_dim, with_global = with_global)
        
        # self.l1_loss = L1Loss()
        self.l1_loss = L1Loss()
        self.ms_ssim_loss = MS_SSIM(data_range = 1.0)
        self.ssim_loss = SSIM(data_range = 1.0)
        # self.perceptual_loss = VGGPerceptualLoss(False)
        # self.tv_loss = TVLoss()


    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        if self.testing:
            images, image_sizes = self.preprocess_test_images(batched_inputs)
            _, _, outputs = self.backbone(images)
            # outputs = self.activation(outputs)
            outputs = self.postprocess_images(outputs, image_sizes)
            return outputs
        
        images, targets, image_sizes = self.preprocess_images(batched_inputs)
        # from IPython import embed
        # from deeplle.utils.image_ops import save_image
        # embed()
        
        _, _, outputs = self.backbone(images)
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
        # loss_dict["perceptual_loss"] = self.perceptual_loss(inputs, targets) * 0.1
        loss_dict["ms_ssim_loss"] = (1 - self.ms_ssim_loss(inputs, targets))
        # loss_dict["tv_loss"] = self.tv_loss(inputs, targets)
        # loss_dict["ssim_loss"] = (1 - self.ssim_loss(inputs, targets))

        return loss_dict

    def preprocess_images(self, batched_inputs: List[Dict[str, Tensor]]):
        images = [self._move_to_current_device(x['image']) for x in batched_inputs]
        targets = [self._move_to_current_device(x['target']) for x in batched_inputs]

        images, image_sizes = processing.pad_collate_images(images, self.size_divisibility)
        targets, _ = processing.pad_collate_images(targets, self.size_divisibility)

        return images, targets, image_sizes

    def preprocess_test_images(self, batched_inputs: List[Dict[str, Tensor]]):
        images = [self._move_to_current_device(x['image']) for x in batched_inputs]
        images, image_sizes = processing.pad_collate_images(images, self.size_divisibility)

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
        images = processing.remove_padding(images, image_sizes)
        return images
