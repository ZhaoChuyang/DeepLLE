# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
import json
import time
import glob
import pathlib
import socket
import torch
from typing import List
import numpy as np


# If you don't set the seed manually, will use current timestamp
# as the seed for random number generator.
_seed = int(time.time())


def read_json(path: str):
    with open(path, "r") as fb:
        data = json.load(fb)
    return data


def find_files(pattern: str):
    """
    Find all files in path based on pattern with filenames matched with pattern,
    return them as a list.

    Args:
        pattern (str): name pattern, such as "*.rar".
    
    Returns:
        A list contains file paths of all files in base directory whose
        path string is matched with pattern.
    """
    return glob.glob(pattern)


def check_path_exists(path: str):
    """
    Return true if the given path exists, false if not.
    """
    return os.path.exists(path)


def check_path_is_image(path: str):
    """
    Check whether the file of the given path is an image by checking the file extension.
    Valid image file extension includes ["jpg", "jpeg", "tiff", "bmp", "png"].
    
    Args:
        path (str): path of the checking file.
    
    Returns:
        Returns True if the file is an image file, otherwise False.
    """
    valid_file_extension = ("jpg", "jpeg", "tiff", "bmp", "png")
    if path.lower().endswith(valid_file_extension):
        return True
    return False


def mkdirs(dirpath: str):
    """
    equivelent to "mkdir -p"
    Make directory in dirpath, if its parent directories do not exist, create its parent directories too.

    Note: this method will not override existing directory.

    Args:
        dirpath (str): path of the target directory.
    """
    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    global _seed
    _seed = int(seed)


def get_seed():
    global _seed
    return _seed


def get_ip_address():
    """
    Get the ip address of current host.
    """
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def convert_to_image(input: torch.Tensor, data_range: List = [0., 1.], input_order="CHW") -> np.ndarray:
    """
    Args:
        input (tensor): input tensor of shape (C, H, W).
    """
    assert isinstance(input, torch.Tensor), "Expect input as Tensor."
    
    assert input_order in ["CHW", "HWC"], "input_order must be chosen from ['CHW', 'HWC']"
    if input_order == "CHW":
        input = torch.einsum("chw->hwc", input)
    
    input = input.detach().cpu().numpy()

    input = (input - data_range[0]) / (data_range[1] - data_range[0])
    input = np.clip(input * 255, 0, 255).astype(np.uint8)
    return input
