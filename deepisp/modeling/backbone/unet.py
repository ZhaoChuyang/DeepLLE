# Created on Tue Oct 11 2022 by Chuyang Zhao
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["UNet"]


"""
Building Blocks of UNet
"""


# class DoubleConv(nn.Module):
#     def __init__(self, in_size, out_size, mid_size=None, relu_slope=0.2, use_HIN=True):
#         super(DoubleConv, self).__init__()
#         if mid_size is None:
#             mid_size = out_size
#         self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

#         self.conv_1 = nn.Conv2d(in_size, mid_size, kernel_size=3, padding=1, bias=False)
#         self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
#         self.conv_2 = nn.Conv2d(mid_size, out_size, kernel_size=3, padding=1, bias=False)
#         self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

#         if use_HIN:
#             # self.norm = nn.GroupNorm(num_groups=1, num_channels=mid_size//2, affine=True)
#             self.norm = nn.InstanceNorm2d(mid_size//2, affine=True)
#             # self.norm = nn.GroupNorm(num_groups=1, num_channels=mid_size, affine=True)

#         self.use_HIN = use_HIN


#     def forward(self, x):
#         out = self.conv_1(x)

#         if self.use_HIN:
#             out_1, out_2 = torch.chunk(out, 2, dim=1)
#             out = torch.cat([self.norm(out_1), out_2], dim=1)
#             # out = self.norm(out) + out
#             # out = torch.cat([self.norm(out), out], dim=1)
#             # out = self.norm(out)
        
#         out = self.relu_1(out)
#         out = self.conv_2(out)
#         out = self.relu_2(out)

#         out += self.identity(x)
#         return out


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2

    input: (b, in_channels, h, w)
    output: (b, out_channels, h, w)
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.InstanceNorm2d(num_features=mid_channels),
            # nn.GroupNorm(num_groups=mid_channels // 4, num_channels=mid_channels),
            # nn.GroupNorm(num_groups=1, num_channels=mid_channels, affine=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            # nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
            # nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv

    input => MaxPool => DoubleConv => output

    input: (b, in_channels, h, w)
    output: (b, out_channels, h // 2, w // 2)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv

    Case 1: Transpose Conv
    x1 => ConvTranspose
    (x1 + x2) => DoubleConv => output

    Case 2: Bilinear
    x1 => Upsample
    (x1 + x2) => DoubleConv => output
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    1x1 conv which will only changes the output channels of the inputs.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


"""
UNet Network
"""

class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 3, bilinear: bool = False, scales: int = 16, base_dim: int = 64):
        """
        Simple UNet implementation.

        Args:
            n_channels (int): num of input channels, typically 3 for images in RGB format.
            n_classes (int): num of output channels, default is 3.
            bilinear (bool): set True to use bilinear + conv to do the upsampling, otherwise
                will use transposed conv to do the upsampling, by default is False.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scales = scales
        
        assert self.scales in [4, 16]
        
        self.inc = DoubleConv(n_channels, base_dim)
        if self.scales == 4:
            self.down1 = Down(base_dim, base_dim * 2)
            factor = 2 if bilinear else 1
            self.down2 = Down(base_dim * 2, base_dim * 4 // factor)
            self.up3 = Up(base_dim * 4, base_dim * 2 // factor, bilinear)
            self.up4 = Up(base_dim * 2, base_dim, bilinear)

        elif self.scales == 16:
            self.down1 = Down(base_dim, base_dim * 2)
            self.down2 = Down(base_dim * 2, base_dim * 4)
            self.down3 = Down(base_dim * 4, base_dim * 8)
            factor = 2 if bilinear else 1
            self.down4 = Down(base_dim * 8, base_dim * 16 // factor)
        
            self.up1 = Up(base_dim * 16, base_dim * 8 // factor, bilinear)
            self.up2 = Up(base_dim * 8, base_dim * 4 // factor, bilinear)
            self.up3 = Up(base_dim * 4, base_dim * 2 // factor, bilinear)
            self.up4 = Up(base_dim * 2, base_dim, bilinear)

        self.outc = OutConv(base_dim, n_classes)
        # self.final = OutConv(n_classes * 2, n_classes)
        # self._initialize()

    
    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): x is of (B, C, H, W), C equals self.n_channels.

        Returns:
            outputs (Tensor): outputs is of (B, C, H, W), C equals self.n_classes,
                B, H and W are the same as input x.
        """
        inp = x
        x1 = self.inc(x)
        if self.scales == 16:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        
        elif self.scales == 4:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
        
        logits = self.outc(x)
        # logits = self.final(torch.cat([inp, out], dim=1))

        # TODO: add activation function to
        # constrain the outputs to valid range
        return logits
