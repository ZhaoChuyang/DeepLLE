# adapted from https://github.com/creotiv/hdrnet-pytorch/blob/master/model.py
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


__all__ = ["HDRPointwiseNN"]


"""
Helper function for HDRNet
"""
import torch

def lerp_weight(x, xs):
  """Linear interpolation weight from a sample at x to xs.
  Returns the linear interpolation weight of a "query point" at coordinate `x`
  with respect to a "sample" at coordinate `xs`.
  The integer coordinates `x` are at pixel centers.
  The floating point coordinates `xs` are at pixel edges.
  (OpenGL convention).
  Args:
    x: "Query" point position.
    xs: "Sample" position.
  Returns:
    - 1 when x = xs.
    - 0 when |x - xs| > 1.
  """
  dx = x - xs
  abs_dx = abs(dx)
  return torch.maximum(torch.tensor(1.0).to(x.device) - abs_dx, torch.tensor(0.0).to(x.device))


def smoothed_abs(x, eps):
  """A smoothed version of |x| with improved numerical stability."""
  return torch.sqrt(torch.multiply(x, x) + eps)


def smoothed_lerp_weight(x, xs):
  """Smoothed version of `LerpWeight` with gradients more suitable for backprop.
  Let f(x, xs) = LerpWeight(x, xs)
               = max(1 - |x - xs|, 0)
               = max(1 - |dx|, 0)
  f is not smooth when:
  - |dx| is close to 0. We smooth this by replacing |dx| with
    SmoothedAbs(dx, eps) = sqrt(dx * dx + eps), which has derivative
    dx / sqrt(dx * dx + eps).
  - |dx| = 1. When smoothed, this happens when dx = sqrt(1 - eps). Like ReLU,
    We just ignore this (in the implementation below, when the floats are
    exactly equal, we choose the SmoothedAbsGrad path since it is more useful
    than returning a 0 gradient).
  Args:
    x: "Query" point position.
    xs: "Sample" position.
    eps: a small number.
  Returns:
    max(1 - |dx|, 0) where |dx| is smoothed_abs(dx).
  """
  eps = torch.tensor(1e-8).to(torch.float32).to(x.device)
  dx = x - xs
  abs_dx = smoothed_abs(dx, eps)
  return torch.maximum(torch.tensor(1.0).to(x.device) - abs_dx, torch.tensor(0.0).to(x.device))

def _bilateral_slice(grid, guide):
    """Slices a bilateral grid using the a guide image.
    Args:
      grid: The bilateral grid with shape (gh, gw, gd, gc).
      guide: A guide image with shape (h, w). Values must be in the range [0, 1].
    Returns:
      sliced: An image with shape (h, w, gc), computed by trilinearly
      interpolating for each grid channel c the grid at 3D position
      [(i + 0.5) * gh / h,
       (j + 0.5) * gw / w,
       guide(i, j) * gd]
    """
    dev = grid.device
    ii, jj = torch.meshgrid(
        [torch.arange(guide.shape[0]).to(dev), torch.arange(guide.shape[1]).to(dev)], indexing='ij')

    scale_i = grid.shape[0] / guide.shape[0]
    scale_j = grid.shape[1] / guide.shape[1]

    gif = (ii + 0.5) * scale_i
    gjf = (jj + 0.5) * scale_j
    gkf = guide * grid.shape[2]

    # Compute trilinear interpolation weights without clamping.
    gi0 = torch.floor(gif - 0.5).to(torch.int32)
    gj0 = torch.floor(gjf - 0.5).to(torch.int32)
    gk0 = torch.floor(gkf - 0.5).to(torch.int32)
    gi1 = gi0 + 1
    gj1 = gj0 + 1
    gk1 = gk0 + 1

    wi0 = lerp_weight(gi0 + 0.5, gif)
    wi1 = lerp_weight(gi1 + 0.5, gif)
    wj0 = lerp_weight(gj0 + 0.5, gjf)
    wj1 = lerp_weight(gj1 + 0.5, gjf)
    wk0 = smoothed_lerp_weight(gk0 + 0.5, gkf)
    wk1 = smoothed_lerp_weight(gk1 + 0.5, gkf)

    w_000 = wi0 * wj0 * wk0
    w_001 = wi0 * wj0 * wk1
    w_010 = wi0 * wj1 * wk0
    w_011 = wi0 * wj1 * wk1
    w_100 = wi1 * wj0 * wk0
    w_101 = wi1 * wj0 * wk1
    w_110 = wi1 * wj1 * wk0
    w_111 = wi1 * wj1 * wk1

    # But clip when indexing into `grid`.
    gi0c = gi0.clip(0, grid.shape[0] - 1).to(torch.long)
    gj0c = gj0.clip(0, grid.shape[1] - 1).to(torch.long)
    gk0c = gk0.clip(0, grid.shape[2] - 1).to(torch.long)

    gi1c = (gi0 + 1).clip(0, grid.shape[0] - 1).to(torch.long)
    gj1c = (gj0 + 1).clip(0, grid.shape[1] - 1).to(torch.long)
    gk1c = (gk0 + 1).clip(0, grid.shape[2] - 1).to(torch.long)

    #        ijk: 0 means floor, 1 means ceil.
    grid_val_000 = grid[gi0c, gj0c, gk0c, :]
    grid_val_001 = grid[gi0c, gj0c, gk1c, :]
    grid_val_010 = grid[gi0c, gj1c, gk0c, :]
    grid_val_011 = grid[gi0c, gj1c, gk1c, :]
    grid_val_100 = grid[gi1c, gj0c, gk0c, :]
    grid_val_101 = grid[gi1c, gj0c, gk1c, :]
    grid_val_110 = grid[gi1c, gj1c, gk0c, :]
    grid_val_111 = grid[gi1c, gj1c, gk1c, :]

    # Append a singleton "channels" dimension.
    w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111 = torch.atleast_3d(
        w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111)

    # TODO(jiawen): Cache intermediates and pass them in.
    # Just pass out w_ijk and the same ones multiplied by by dwk.
    return (torch.multiply(w_000, grid_val_000) +
            torch.multiply(w_001, grid_val_001) +
            torch.multiply(w_010, grid_val_010) +
            torch.multiply(w_011, grid_val_011) +
            torch.multiply(w_100, grid_val_100) +
            torch.multiply(w_101, grid_val_101) +
            torch.multiply(w_110, grid_val_110) +
            torch.multiply(w_111, grid_val_111))

@torch.jit.script
def batch_bilateral_slice(grid, guide):
    res = []
    for i in range(grid.shape[0]):
        res.append(_bilateral_slice(grid[i], guide[i]).unsqueeze(0))
    return torch.concat(res, 0)

def trace_bilateral_slice(grid, guide):
    return batch_bilateral_slice(grid, guide)


# grid: The bilateral grid with shape (gh, gw, gd, gc).
# guide: A guide image with shape (h, w). Values must be in the range [0, 1].

grid = torch.rand(1, 3, 3, 8, 12).cuda()
guide = torch.rand(1,16, 16).cuda()

bilateral_slice = torch.jit.trace(
    trace_bilateral_slice, (grid, guide))

bilateral_slice(grid, guide)


class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

        if use_bias and not batch_norm:
            self.conv.bias.data.fill_(0.00)
        # aka TF variance_scaling_initializer
        torch.nn.init.kaiming_uniform_(self.conv.weight)#, mode='fan_out',nonlinearity='relu')
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc , outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None  
        
        if not batch_norm:
            self.fc.bias.data.fill_(0.00)
        # aka TF variance_scaling_initializer
        torch.nn.init.kaiming_uniform_(self.fc.weight)#, mode='fan_out',nonlinearity='relu')
        
    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        bilateral_grid = bilateral_grid.permute(0,3,4,2,1)
        guidemap = guidemap.squeeze(1)
        # grid: The bilateral grid with shape (gh, gw, gd, gc).
        # guide: A guide image with shape (h, w). Values must be in the range [0, 1].
        coeefs = bilateral_slice(bilateral_grid, guidemap).permute(0,3,1,2)
        return coeefs


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        # out_channels = []
        # for chan in range(n_out):
        #     ret = scale[:, :, :, chan, 0]*input_image[:, :, :, 0]
        #     for chan_i in range(1, n_in):
        #         ret += scale[:, :, :, chan, chan_i]*input_image[:, :, :, chan_i]
        #     if has_affine_term:
        #         ret += offset[:, :, :, chan]
        #     ret = tf.expand_dims(ret, 3)
        #     out_channels.append(ret)

        # ret = tf.concat(out_channels, 3)
        """
            R = r1[0]*r2 + r1[1]*g2 + r1[2]*b3 +r1[3]
        """

        # print(coeff.shape)
        # R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        # G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        # B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 9:10, :, :]
        G = torch.sum(full_res_input * coeff[:, 3:6, :, :], dim=1, keepdim=True) + coeff[:, 10:11, :, :]
        B = torch.sum(full_res_input * coeff[:, 6:9, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, params['guide_complexity'], kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(params['guide_complexity'], 1, kernel_size=1, padding=0, activation= nn.Sigmoid) #nn.Tanh nn.Sigmoid

    def forward(self, x):
        return self.conv2(self.conv1(x))#.squeeze(1)


class Coeffs(nn.Module):

    def __init__(self, nin=4, nout=3, params=None):
        super(Coeffs, self).__init__()
        self.params = params
        self.nin = nin 
        self.nout = nout
        
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']
        bn = params['batch_norm']
        nsize = params['net_input_size']

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize/sb))
        self.splat_features = nn.ModuleList()
        prev_ch = 3
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm*(2**i)*lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm*(2**i)*lb

        # global features
        n_layers_global = int(np.log2(sb/4))
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(ConvBlock(prev_ch, cm*8*lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm*8*lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize/2**n_total)**2
        self.global_features_fc.append(FC(prev_ch, 32*cm*lb, batch_norm=bn))
        self.global_features_fc.append(FC(32*cm*lb, 16*cm*lb, batch_norm=bn))
        self.global_features_fc.append(FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8*cm*lb, 3, batch_norm=bn))
        self.local_features.append(ConvBlock(8*cm*lb, 8*cm*lb, 3, activation=None, use_bias=False))
        
        # predicton
        self.conv_out = ConvBlock(8*cm*lb, lb*nout*nin, 1, padding=0, activation=None)#,batch_norm=True)

   
    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params['luma_bins']
        cm = params['channel_multiplier']
        sb = params['spatial_bin']

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x
        
        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)        
        local_features = x

        fusion_grid = local_features
        fusion_global = global_features.view(bs,8*cm*lb,1,1)
        fusion = self.relu( fusion_grid + fusion_global )

        x = self.conv_out(fusion)
        s = x.shape
        y = torch.stack(torch.split(x, self.nin*self.nout, 1),2)
        # y = torch.stack(torch.split(y, self.nin, 1),3)
        # print(y.shape)
        # x = x.view(bs,self.nin*self.nout,lb,sb,sb) # B x Coefs x Luma x Spatial x Spatial
        # print(x.shape)
        return y


class HDRPointwiseNN(nn.Module):

    def __init__(self, params):
        super(HDRPointwiseNN, self).__init__()
        self.coeffs = Coeffs(params=params)
        self.guide = GuideNN(params=params)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        # self.bsa = bsa.BilateralSliceApply()

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        # out = bsa.bsa(coeffs,guide,fullres)
        return out