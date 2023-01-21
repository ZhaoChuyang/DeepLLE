from torch import nn
import torch
from torch.nn import functional as F
from .spynet import SpyNet
from .nn_utils import default_init_weights, flow_warp


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.
    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        resnet = nn.Sequential(*[ResidualBlockNoBN(num_out_ch) for _ in range(num_block)])
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), 
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            resnet
        )

    def forward(self, fea):
        return self.main(fea)


class RecurrentVideoModel(nn.Module):
    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward
    
    def forward(self, x):
        """Forward function of BasicVSR.
        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            from IPython import embed
            embed()
            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1) # (b, 2 * num_feat, h, w)
            out = self.lrelu(self.fusion(out)) # (b, num_feat, h, w)
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out))) # (b, num_feat, 2 * h, 2 * w)
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out))) # (b, 64, 4 * h, 4 * w)
            out = self.lrelu(self.conv_hr(out)) # (b, 64, 4 * h, 4 * w)
            out = self.conv_last(out) # (b, 3, 4 * h, 4 * w)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False) # (b, 3, 4 * h, 4 * w)
            out += base # (b, 3, 4 * h, 4 * w)
            out_l[i] = out # (b, 3, 4 * h, 4 * w)

        return torch.stack(out_l, dim=1)


if __name__ == '__main__':
    model = RecurrentVideoModel()
    model.eval()
    x = torch.rand(1, 7, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        print(out.shape)