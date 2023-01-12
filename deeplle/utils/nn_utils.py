from torch import nn
from typing import Tuple
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.optim import Optimizer


def get_bare_model(model) -> nn.Module:
    """
    Get the bare model from a wrapped model.

    Args:
        model (nn.Module): model to be resumed.

    Returns:
        nn.Module: bare model
    """
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        return model.module
    return model


def get_model_info(model) -> Tuple[str, int]:
    """
    Get the model information and parameter number.

    Args:
        model (nn.Module): model to be printed.

    Returns:
        str: model information
        int: number of parameters
    """
    model = get_bare_model(model)
    model_str = str(model)
    model_params = sum(map(lambda x: x.numel(), model.parameters()))
    return model_str, model_params


def get_learning_rate(optimizer: Optimizer) -> float:
    """
    Get the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): optimizer 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
