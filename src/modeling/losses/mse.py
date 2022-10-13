from torch import Tensor
from torch import nn
from torch.nn import functional as F



class L1Loss(nn.Module):
    def __init__(self, reduction: str='mean'):
        """
        Overrided L1 Loss, which supports mask. If mask is not provided,
        it works the same as normal nn.L1Loss.

        Args:
            reduction (str): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        """
        super().__init__()
        self.reduction = reduction
        

    def forward(self, input: Tensor, target: Tensor, mask: Tensor = None):
        if mask is None:
            return F.l1_loss(input, target, reduction=self.reduction)
        
        else:
            loss = F.l1_loss(input, target, reduction='none')
            loss = loss * mask

            if self.reduction == 'mean':
                return loss.sum() / mask.sum()

            if self.reduction == 'sum':
                return loss.sum()

            if self.reduction == 'none':
                return loss

