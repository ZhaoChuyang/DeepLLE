from typing import Dict, Callable
from functools import partial
from deepisp.utils import Registry


METRIC_REGISTRY = Registry("METRIC")
METRIC_REGISTRY.__doc__ = """
Regsitry for all metrics.
"""


def build_metric(name: str, args: Dict) -> Callable:
    """
    Args:
        name (str): metric name.
        args (dict): dict of the keyword arguments.

    Returns:
        metric_fn (callable): metric function takes two inputs,
            the output image and the ground truth image.
    """
    metric_fn = METRIC_REGISTRY.get(name)
    metric_fn = partial(metric_fn, **args)
    return metric_fn
