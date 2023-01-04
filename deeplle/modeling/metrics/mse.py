import torch
import numpy as np
from deeplle.modeling.metrics.build import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def MSE(img1, img2):
    """
    Calculate Mean Squared Error (MSE).

    Args:
        img_1 (np.array): image in range [0, 255].
        img_2 (np.array): image in range [0, 255].
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')

    # convert inputs to numpy arrary and reshape to (H, W, C) if needed.
    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    return np.square(np.subtract(img2, img1)).mean()
