# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
from deeplle.utils import read_json
from ast import literal_eval
from typing import Dict, List
import torch
import copy


__all__ = ["init_config"]


class Config:
    """
    Points to the root of the config directory.
    """
    CONFIG_DIR: str = None


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
    Config.CONFIG_DIR = os.path.dirname(args.config)

    # Step 2: merge with base config.
    if "base" in config:
        config = _merge_base_config(config)

    # Step 3: merge with command line options.
    if args.opts:
        config = _merge_config_with_opts(config, args.opts)

    # Step 4: computing and updating configurations
    config["model"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    config["trainer"]["save_dir"] = os.path.join(Config.CONFIG_DIR, config["trainer"]["save_dir"])
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

    return config


def _merge_base_config(config: Dict) -> Dict:
    """
    Recursively merge current config with its base config.
    """
    if not config.get("base", None):
        return config

    base_config = read_json(os.path.join(Config.CONFIG_DIR, config["base"]))
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