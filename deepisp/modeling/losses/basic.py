import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


__all__ = ["L1Loss", "MSELoss", "CharbonnierLoss", "TVLoss"]


class L1Loss(nn.Module):
    def __init__(self, reduction: str='mean'):
        """
        Extended L1 Loss that supports mask.
    
        L1 Loss is less sensitive to outlier than MSE Loss and in some cases
        prevents exploding gradients (e.g. see the paper Fast R-CNN).

        Args:
            reduction (str): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        """
        super(L1Loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input: Tensor, target: Tensor, mask: Tensor = None):
        """
        Args:
            input (tensor): predicted tensor of shape (b, c, h, w).
            target (tensor): ground truth tensor of shape (b, c, h, w).
            mask (tensor | none): mask for the activated neurons in computation. Default is None.
        """
        if mask is None:
            return F.l1_loss(input, target, reduction=self.reduction)
        
        else:
            assert input.shape == target.shape == mask.shape, "Expect the shapes of input, target and mask match the same."
            loss = F.l1_loss(input, target, reduction='none')
            loss = loss * mask

            if self.reduction == 'mean':
                return loss.mean()

            if self.reduction == 'sum':
                return loss.sum()

            if self.reduction == 'none':
                return loss


class MSELoss(nn.Module):
    def __init__(self,  reduction: str = "mean"):
        """
        Mean Square Loss (L2 Loss).

        Args:
            reduction (str): Specify the reduction to apply to the output.
                Supported choices are "none" | "mean" | "sum". 
        """
        super(MSELoss, self).__init__()
        self.reduction = reduction
        
    
    def forward(self, input: Tensor, target: Tensor, mask: Tensor = None):
        """
        Args:
            input (tensor): predicted tensor of shape (b, c, h, w).
            target (tensor): ground truth tensor of shape (b, c, h, w).
            mask (tensor | none): mask for the activated neurons in computation. Default is None.
        """
        if mask is None:
            return F.mse_loss(input, target, reduction=self.reduction)
        
        else:
            assert input.shape == target.shape == mask.shape, "Expect the shapes of input, target and mask match the same."
            loss = F.mse_loss(input, target, reduction='none')
            loss = loss * mask

            if self.reduction == 'mean':
                return loss.mean()

            if self.reduction == 'sum':
                return loss.sum()

            if self.reduction == 'none':
                return loss


class CharbonnierLoss(nn.Module):
    def __init__(self, reduction: str = "mean", eps: float = 1e-12):
        """
        Charbonnier Loss, a variant of the L1 Loss, more robust than
        the vanilla version.

        Args:
            reduction (str): Specify the reduction to apply to the output.
                Supported choices are "none" | "mean" | "sum".
            eps (float):  A value used to control the curvature near zero. Default: 1e-12.
        """
        super(CharbonnierLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, mask: Tensor = None):
        """
        Args:
            input (tensor): predicted tensor of shape (b, c, h, w).
            target (tensor): ground truth tensor of shape (b, c, h, w).
            mask (tensor | none): mask for the activated neurons in computation. Default is None.
        """
        assert input.shape == target.shape, "Expect the shapes of input and target match the same."
        loss = torch.sqrt((input - target)**2 + self.eps)
        
        if mask is not None:
            assert input.shape == target.shape == mask.shape, "Expect the shapes of input, target and mask match the same."
            loss = loss * mask
        
        if self.reduction == 'mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        if self.reduction == 'none':
            return loss


class TVLoss(L1Loss):
    def __init__(self, reduction: str = "mean"):
        """
        Total variation loss (TV Loss) computes the gradients of pixels along
        x axis and y axis and encourges spatial smoothness in the image.

        Args:
            reduction (str): Specify the reduction to apply to the output.
                Supported choices are "mean" | "sum".
        """
        assert reduction in ["mean", "sum"], "Supported reduction mode is 'mean' | 'sum'."
        super(TVLoss, self).__init__(reduction=reduction)

    def forward(self, input: Tensor):
        y_diff = super().forward(input[:,:,:-1,:], input[:,:,1:,:])
        x_diff = super().forward(input[:,:,:,:-1], input[:,:,:,1:])

        loss = x_diff + y_diff
        return loss
