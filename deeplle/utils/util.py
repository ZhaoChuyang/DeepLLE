# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
import json
import time
import glob
import pathlib
import socket
from typing import List


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


def get_ip_address():
    """
    Get the ip address of current host.
    """
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address
