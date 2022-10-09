# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
from . import read_json
from ast import literal_eval


__all__ = ["init_config"]


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
    1. Read config from config file
    2. Merge config with the options passed by command line.
    3. Add other extra config options
    
    """

    if not args.config:
        raise RuntimeError("config file must be provided")

    config = read_json(args.config)
    
    if args.opts:
        merge_config(config, args.opts)
    
    config["trainer"]["ckp_dir"] = os.path.join(config["trainer"]["save_dir"], config["name"], "checkpoints")
    config["trainer"]["log_dir"] = os.path.join(config["trainer"]["save_dir"], config["name"], "log")

    return config


def merge_config(config: dict, opts: list) -> dict:
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
        # if the key does not exist in the config, create new item with this key in the config dict
        dict_nodes = key.split(".")
        cur_node = config
        for node in dict_nodes[:-1]:
            if node not in cur_node:
                cur_node[node] = dict()
            cur_node = cur_node[node]
        
        cur_node[dict_nodes[-1]] = value

    return config
