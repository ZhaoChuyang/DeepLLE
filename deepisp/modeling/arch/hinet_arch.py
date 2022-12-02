# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
'''
HINet: Half Instance Normalization Network for Image Restoration
@inproceedings{chen2021hinet,
  title={HINet: Half Instance Normalization Network for Image Restoration},
  author={Liangyu Chen and Xin Lu and Jie Zhang and Xiaojie Chu and Chengpeng Chen},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}
'''
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .base import BaseISPModel
from .. import processing
from .build import MODEL_REGISTRY
from ..losses import L1Loss, MS_SSIM, SSIM

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


@MODEL_REGISTRY.register()
class HINet(BaseISPModel):
    def __init__(self, in_chn=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4, testing=False):
        super(HINet, self).__init__()
        self.testing = testing
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_csff=downsample, use_HIN=use_HIN))
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)

        self.last = conv3x3(prev_channels, in_chn, bias=True)

        self.l1_loss = L1Loss()
        self.ms_ssim_loss = MS_SSIM(data_range = 1.0)
        self.ssim_loss = SSIM(data_range = 1.0)

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

    @torch.no_grad()
    def metrics(self, inputs: Tensor, targets: Tensor):
        output_dict = {}
        return output_dict

    def losses(self, inputs: Tensor, targets: Tensor):
        out_1, out_2 = inputs
        assert out_1.shape == targets.shape, "Shapes of out_1: {} and targets: {} do not match.".format(out_1.shape, targets.shape)
        assert out_2.shape == targets.shape, "Shapes of out_2: {} and targets: {} do not match.".format(out_2.shape, targets.shape)
        loss_dict = {}

        loss_dict["l1_loss_1"] = self.l1_loss(out_1, targets)
        loss_dict["l1_loss_2"] = self.l1_loss(out_2, targets)
        # loss_dict["ms_ssim_loss"] = self.ms_ssim_loss(inputs, targets)
        # loss_dict["ssim_loss"] = (1 - self.ssim_loss(inputs, targets)) * 0.2

        return loss_dict

    def forward(self, batched_inputs: List[Dict[str, Tensor]]):
        if self.testing:
            images, image_sizes = self.preprocess_test_images(batched_inputs)
        else:
            images, targets, image_sizes = self.preprocess_images(batched_inputs)

        #stage 1
        x1 = self.conv_01(images)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)

        sam_feature, out_1 = self.sam12(x1, images)
        #stage 2
        x2 = self.conv_02(images)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]))

        out_2 = self.last(x2)
        out_2 = out_2 + images

        if self.testing:
            outputs = self.postprocess_images(out_2, image_sizes)
            return outputs

        outputs = [out_1, out_2]
        loss_dict = self.losses(outputs, targets)
        output_dict = self.metrics(outputs, targets)

        return loss_dict, output_dict

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
    
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


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc


if __name__ == "__main__":
    pass