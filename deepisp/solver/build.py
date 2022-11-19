# Created on Mon Oct 10 2022 by Chuyang Zhao
import torch
from .lr_scheduler import WarmupMultiStepLR, WarmupCosineLR


def get_default_optimizer_params(
    model,
    weight_decay_norm = None,
):
    """
    If you want to disable some layers for training, set 
    requires_grad of their parameters to False in the
    model wrapper.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params = []
    memo = set()

    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            
            # Avoid duplicating parameters, which happens when some modules share the same sub module.
            if value in memo:
                continue
            memo.add(value)

            hyperparams = {}

            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm

            params.append({"params": [value], **hyperparams})
    
    # TODO: speed up multi-tensor optimizer by merging duplicated groups,
    # as introduced in: https://github.com/facebookresearch/detectron2/blob/main/detectron2/solver/build.py
    return params


def build_optimizer(model, name, weight_decay_norm = None, **kwargs):
    params = get_default_optimizer_params(model, weight_decay_norm)
    optimizer = getattr(torch.optim, name)(params=params, **kwargs)
    return optimizer


def build_lr_scheduler(optimizer: torch.optim.Optimizer, name: str, **kwargs):
    if name == "WarmupMultiStepLR":
        scheduler = WarmupMultiStepLR(optimizer, **kwargs)
    elif name == "WarmupCosineLR":
        scheduler = WarmupCosineLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(
            "LRScheduler: {} is not implemented. Please choose from \"WarmupMultiStepLR\" and \"WarmupCosineLR\".".format(name))
    return scheduler