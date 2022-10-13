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
    args = cfg_model["args"]
    
    assert "testing" in args, (
        f"'testing' needs to be explicitly specified in config, " \
        "but was not found in model's args: {args}" \
        "set it to False if you are training model, True if you are evaluating or tracing model"
    )
    
    model = MODEL_REGISTRY.get(name)(**args)
    return model
