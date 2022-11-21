# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
from . import read_json
from ast import literal_eval
from typing import Dict, List
import copy


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
    1. Read config from config file.
    2. Merge config from step 1 with the base config.
    3. Merge config from step 2 with the options specified in command line.
    4. Add other extra config options.
    
    """

    if not args.config:
        raise RuntimeError("config file must be provided")

    # Step 1: read config.
    config = read_json(args.config)

    # Step 2: merge with base config.
    # NOTE: We only merge base config at depth-1. Currently recursive config merging is not supported.
    if "base" in config:
        base_config = read_json(config["base"])
        config = _merge_config_with_base(config, base_config)

    # Step 3: merge with command line options.
    if args.opts:
        config = _merge_config_with_opts(config, args.opts)
    
    # Step 4: add other configurations
    config["trainer"]["ckp_dir"] = os.path.join(config["trainer"]["save_dir"], config["name"], "checkpoints")
    config["trainer"]["log_dir"] = os.path.join(config["trainer"]["save_dir"], config["name"], "log")

    return config


def _merge_config_with_base(config: Dict, base: Dict) -> Dict:
    """
    Merge current config with base config.
    Items exist in current config but not in base config will be kept in the merged config.
    Items exist in both current config and base config will keep the version of current config.
    Items not exist in current config but exist in base config will keep the version of base config.
    
    Args:
        config (dict): json dict of current config.
        base (dict): json dict of base config.
    """
    if base is not None:
        merged = copy.deepcopy(base)
    else:
        merged = {}
    
    for key in config.keys():
        if key not in merged:
            merged[key] = copy.deepcopy(config[key])
        else:
            if isinstance(config[key], Dict):
                merged[key] = _merge_config_with_base(config[key], merged[key])
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
