# Created on Mon Oct 10 2022 by Chuyang Zhao
from ...utils import Registry


MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for the whole model.

The registry object will be called with `obj(**cfg_model)`,
and expected to return a nn.Module.
"""


def build_model(cfg_model):
    """
    Build the whole model, arguments for building the model
    should be defined in `cfg_model`. 
    """
    name = cfg_model["name"]
    model = MODEL_REGISTRY.get(name)(**cfg_model)
    return model
