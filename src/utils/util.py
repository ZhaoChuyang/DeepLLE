# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
import json
import time
import pathlib


# If you don't set the seed manually, will use current timestamp
# as the seed for random number generator.
_seed = int(time.time())


def read_json(path: str):
    with open(path, "r") as fb:
        data = json.load(fb)
    return data


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
