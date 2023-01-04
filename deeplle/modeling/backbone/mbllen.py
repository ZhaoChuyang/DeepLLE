import torch
from torch import nn


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True)
    )

def conv2dtranspose(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU(inplace=True)
    )


class MBLLEN(nn.Module):
    def __init__(self):
        """
        Adapted from https://github.com/Lvfeifan/MBLLEN
        """
        super().__init__()

        self.FEM = conv2d(3, 32, 3, 1, 1)
        self.EM_com = self.EM(kernel_size=5, in_channels=32, channel=8)

        self.layer1 = self.FEM_EM(32)
        self.layer2 = self.FEM_EM(3)
        self.layer3 = self.FEM_EM(3)
        self.layer4 = self.FEM_EM(3)
        self.layer5 = self.FEM_EM(3)
        self.layer6 = self.FEM_EM(3)
        self.layer7 = self.FEM_EM(3)
        self.layer8 = self.FEM_EM(3)
        self.layer9 = self.FEM_EM(3)

        # input_channels = (num_layers+1) * 3 = 30 
        self.out_conv = conv2d(30, 3, 1, 1, 0)

    def EM(self, kernel_size, in_channels, channel):
        EM_Block = nn.Sequential(
            conv2d(in_channels, channel, 3, 1, 1),
            conv2d(channel, channel, kernel_size, 1, 0),
            conv2d(channel, channel*2, kernel_size, 1, 0),
            conv2d(channel*2, channel*4, kernel_size, 1, 0),
            conv2dtranspose(channel*4, channel*2, kernel_size, 1, 0),
            conv2dtranspose(channel*2, channel, kernel_size, 1, 0),
            conv2dtranspose(channel, 3, kernel_size, 1, 0),
        )
        return EM_Block

    def FEM_EM(self, in_channels=3):
        return nn.Sequential(
            conv2d(in_channels, 32, 3, 1, 1),
            self.EM(kernel_size=5, in_channels=32, channel=8)
        )

    def forward(self, x):
        x = self.FEM(x) # (b, 32, h, w)
        em_con = self.EM_com(x) # (b, 3, h, w)
 
        x = self.layer1(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 6, h, w)
        x = self.layer2(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 9, h, w)
        x = self.layer3(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 12, h, w)
        x = self.layer4(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 15, h, w)
        x = self.layer5(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 18, h, w)
        x = self.layer6(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 21, h, w)
        x = self.layer7(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 24, h, w)
        x = self.layer8(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 27, h, w)
        x = self.layer9(x) # (b, 3, h, w)
        em_con = torch.cat([em_con, x], dim=1) # (b, 30, h, w)

        out = self.out_conv(em_con) # (b, 3, h, w)
        return out
