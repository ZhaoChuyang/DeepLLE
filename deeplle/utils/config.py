# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
from ast import literal_eval
from typing import Dict, List
import torch
import copy
import inspect
import functools
from deeplle.utils import read_json


__all__ = ["init_config", "configurable", "ConfigDict"]


_CONFIG_DIR: str = None


class ConfigDict(dict):
    """
    ConfigDict modified from addict (https://github.com/mewwts/addict/blob/master/addict/addict.py).

    Convert the dict to a class with attributes, so you can access the values either by `__getitem__`
    or `__getattr__`.

    Examples:
        >>> val = cfg["a"]["b"]
        >>> val = cfg.a.b
    The above two ways are equivalent.
    """
    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in kwargs.items():
            __self[key] = __self._hook(val)

    def __setattr__(self, name, value):
        if hasattr(self.__class__, name):
            raise AttributeError("'Dict' object attribute "
                                 "'{0}' is read-only".format(name))
        else:
            self[name] = value

    def __setitem__(self, name, value):
        isFrozen = (hasattr(self, '__frozen') and
                    object.__getattribute__(self, '__frozen'))
        if isFrozen and name not in super(ConfigDict, self).keys():
                raise KeyError(name)
        super(ConfigDict, self).__setitem__(name, value)
        try:
            p = object.__getattribute__(self, '__parent')
            key = object.__getattribute__(self, '__key')
        except AttributeError:
            p = None
            key = None
        if p is not None:
            p[key] = self
            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')

    def __add__(self, other):
        if not self.keys():
            return other
        else:
            self_type = type(self).__name__
            other_type = type(other).__name__
            msg = "unsupported operand type(s) for +: '{}' and '{}'"
            raise TypeError(msg.format(self_type, other_type))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __missing__(self, name):
        if object.__getattribute__(self, '__frozen'):
            raise KeyError(name)
        # return self.__class__(__parent=self, __key=name)
        raise KeyError(f"'{self.__class__.__name__}' object has no key '{name}'")

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        base = {}
        for key, value in self.items():
            if isinstance(value, type(self)):
                base[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                base[key] = type(value)(
                    item.to_dict() if isinstance(item, type(self)) else
                    item for item in value)
            else:
                base[key] = value
        return base

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def update(self, *args, **kwargs):
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError()
            other.update(args[0])
        other.update(kwargs)
        for k, v in other.items():
            if ((k not in self) or
                (not isinstance(self[k], dict)) or
                (not isinstance(v, dict))):
                self[k] = v
            else:
                self[k].update(v)

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def __or__(self, other):
        if not isinstance(other, (ConfigDict, dict)):
            return NotImplemented
        new = ConfigDict(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, (ConfigDict, dict)):
            return NotImplemented
        new = ConfigDict(other)
        new.update(self)
        return new

    def __ior__(self, other):
        self.update(other)
        return self

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def freeze(self, shouldFreeze=True):
        object.__setattr__(self, '__frozen', shouldFreeze)
        for key, val in self.items():
            if isinstance(val, ConfigDict):
                val.freeze(shouldFreeze)

    def unfreeze(self):
        self.freeze(False)


def _convert_from_string(string: str):
    """Convert string to appropriate type automatically
    Valid data types include int, float, string, list, dict.
    Args:
        string: input string to convert
    
    Returns:
        If conversion is successful, return data that converted from string of appropriate type,
        otherwise return the original string.
    """
    try:
        return literal_eval(string)
    except:
        return string


def init_config(args):
    """Initialize the config
    Initialization is composed of two steps:
    1. Read config from config file.
    2. Merge config from step 1 with the base config.
    3. Merge config from step 2 with the options specified in command line.
    4. Add other extra config options.
    
    """
    if not args.config:
        raise RuntimeError("config file must be provided")

    # Step 1: read config.
    global _CONFIG_DIR
    config = read_json(args.config)
    _CONFIG_DIR = os.path.abspath(os.path.dirname(args.config))

    # Step 2: merge with base config.
    if "base" in config:
        config = _merge_base_config(config)

    # Step 3: merge with command line options.
    if args.opts:
        config = _merge_config_with_opts(config, args.opts)

    # Step 4: updating configurations
    config["model"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    config["trainer"]["save_dir"] = os.path.join(_CONFIG_DIR, config["trainer"]["save_dir"])
    config["trainer"]["ckp_dir"] = os.path.join(config["trainer"]["save_dir"], config["name"], "checkpoints")
    config["trainer"]["log_dir"] = os.path.join(config["trainer"]["save_dir"], config["name"], "log")

    if config["trainer"]["resume_checkpoint"] and not os.path.isabs(config["trainer"]["resume_checkpoint"]):
        config["trainer"]["resume_checkpoint"] = os.path.join(config["trainer"]["ckp_dir"], config["trainer"]["resume_checkpoint"])
    
    if config["test"]["resume_checkpoint"] and not os.path.isabs(config["test"]["resume_checkpoint"]):
        config["test"]["resume_checkpoint"] = os.path.join(config["trainer"]["ckp_dir"], config["test"]["resume_checkpoint"])

    if config["infer"]["resume_checkpoint"] and not os.path.isabs(config["infer"]["resume_checkpoint"]):
        config["infer"]["resume_checkpoint"] = os.path.join(config["trainer"]["ckp_dir"], config["infer"]["resume_checkpoint"])
    
    if not os.path.isabs(config["infer"]["save_dir"]):
        config["infer"]["save_dir"] = os.path.join(config["trainer"]["save_dir"], config["name"], "test", config["infer"]["save_dir"])

    return ConfigDict(config)


def _merge_base_config(config: Dict) -> Dict:
    """
    Recursively merge current config with its base config.
    """
    if not config.get("base", None):
        return config
    
    global _CONFIG_DIR
    base_config = read_json(os.path.join(_CONFIG_DIR, config["base"]))
    base_config = _merge_base_config(base_config)
    
    merged_config = _merge_dicts(config, base_config)
    return merged_config


def _merge_dicts(config: Dict, base: Dict) -> Dict:
    """
    Merge current config with base config.
    Items exist in current config but not in base config will be kept in the merged config.
    Items exist in both current config and base config will keep the version of current config.
    Items not exist in current config but exist in base config will keep the version of base config.
    
    Args:
        config (dict): json dict of current config.
        base (dict): json dict of base config.
    """
    if base is None:
        return config

    merged = copy.deepcopy(base)
    
    for key in config.keys():
        if key not in merged:
            merged[key] = copy.deepcopy(config[key])
        else:
            if isinstance(config[key], Dict):
                merged[key] = _merge_dicts(config[key], merged[key])
            else:
                merged[key] = copy.deepcopy(config[key])
    
    return merged


def _merge_config_with_opts(config: Dict, opts: List) -> Dict:
    """Merge existing config with command line options
    Args:
        config: (dict) Parsed json object.
        opts: (list[str]) Extra config options used to modify config.
        If the option has already been in the config, modify the existed item,
        otherwise create new option item in the config dict.
    Returns:
        Modified config dict.
    """

    p = 0
    while p < len(opts):
        # option of format "key=value"
        if "=" in opts[p]:
            key, value = opts[p].split("=")
            p += 1
        # option of format "key value"
        else:
            key = opts[p]
            value = opts[p+1]
            p += 2
        
        value = _convert_from_string(value)

        # locate the option key in config to modify
        # if the key does not exist in the config, raise RuntimeError
        dict_nodes = key.split(".")
        cur_node = config
        for node in dict_nodes[:-1]:
            if node not in cur_node:
                # cur_node[node] = dict()
                raise RuntimeError(f"config node: {node} does not exist in config file, check its name or add it to config file first.")
            cur_node = cur_node[node]
        
        if isinstance(cur_node[dict_nodes[-1]], value):
            cur_node[dict_nodes[-1]] = value
        else:
            raise RuntimeError(f"{key} expects {type(cur_node[dict_nodes[-1]])}. Got {type(value)}.")

    return config


def configurable(init_func=None, *, from_config=None):
    """
    Modified from detectron2 (https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/config.py).
    
    Decorate a function or a class's __init__ method so that it can be called
    with a :class:`ConfigDict` object using a :func:`from_config` function that translates
    :class:`ConfigDict` to arguments.
    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            @classmethod
            def from_config(cls, cfg):   # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}
        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg: {"a": cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass
        a1 = a_func(a=1, b=2)  # regular call
        a2 = a_func(cfg)       # call with a cfg
        a3 = a_func(cfg, b=3, c=4)  # call with extra overwrite
    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The
            class must have a ``from_config`` classmethod which takes `cfg` as
            the first argument.
        from_config (callable): the from_config function in usage 2. It must take `cfg`
            as its first argument.
    """

    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a 'from_config' classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    else:
        if from_config is None:
            return configurable  # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)

            wrapped.from_config = from_config
            return wrapped

        return wrapper


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take 'cfg' as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    if len(args) and isinstance(args[0], ConfigDict):
        return True
    if isinstance(kwargs.pop("cfg", None), ConfigDict):
        return True
    # `from_config`'s first argument is forced to be "cfg".
    # So the above check covers all cases.
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Config parser")
    parser.add_argument("--config", type=str)
    parser.add_argument("--opts", default=None)
    args = parser.parse_args()
    config = init_config(args)
    print(config)