# Created on Tue Oct 11 2022 by Chuyang Zhao
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["UNet"]


"""
Building Blocks of UNet
"""

class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2

    input: (b, in_channels, h, w)
    output: (b, out_channels, h, w)
    """

    def __init__(self, in_channels, out_channels, mid_channels: int = None, residual: bool = False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False) if residual else None
        
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
        if self.res_conv:
            return self.double_conv(x) + self.res_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv

    input => MaxPool => DoubleConv => output

    input: (b, in_channels, h, w)
    output: (b, out_channels, h // 2, w // 2)
    """

    def __init__(self, in_channels: int, out_channels: int, residual: bool = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, residual)
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

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, residual: bool = False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, residual)
            # self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, residual)

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
    def __init__(self, n_channels: int = 3, n_classes: int = 3, bilinear: bool = False, base_dim: int = 64, depth: int = 4, residual: bool = False):
        """
        Simple UNet implementation.

        Args:
            n_channels (int): num of input channels, typically 3 for images in RGB format.
            n_classes (int): num of output channels, default is 3.
            bilinear (bool): set True to use bilinear + conv to do the upsampling, otherwise
                will use transposed conv to do the upsampling, by default is False.
            base_dim (int): base dimension of the layers of UNet, 64 by default.
            depth (int): depth of UNet, 4 by default.
            residual (bool): whether use residual connection in the basic conv block in UNet, do not use by default.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, base_dim)

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.mid_layers = nn.ModuleList()
        
        factor = 2 if bilinear else 1
        prev_dim = base_dim

        # 1,2..depth (num: depth)
        for i in range(1, depth+1):
            if i == depth:
                self.down_layers.append(Down(prev_dim, base_dim * (2**i) // factor, residual))
            else:
                self.down_layers.append(Down(prev_dim, base_dim * (2**i), residual))
            prev_dim = base_dim * (2**i)

        # depth-1..1,0 (num: depth)
        for i in range(depth-1, -1, -1):
            if i == 0:
                self.up_layers.append(Up(prev_dim, base_dim * (2**i), bilinear, residual))
            else:
                self.up_layers.append(Up(prev_dim, base_dim * (2**i) // factor, bilinear, residual))
            prev_dim = base_dim * (2**i)


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
        x = self.inc(x)
        stack = []

        for down in self.down_layers:
            stack.append(x)
            x = down(x)
            
        for up in self.up_layers:
            x_prev = stack.pop()
            x = up(x, x_prev)
        
        logits = self.outc(x)

        # TODO: add activation function to
        # constrain the outputs to valid range
        return logits
