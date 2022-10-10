# Created on Sat Oct 08 2022 by Chuyang Zhao
import os
import json
import time
import pathlib


# If you don't set the seed manually, will use current timestamp
# as the seed for random number generator.
_seed = int(time.time())


def read_json(path):
    with open(path, "r") as fb:
        data = json.load(fb)
    return data


def check_path_exists(path):
    """Return true if the given path exists, false if not.
    """
    return os.path.exists(path)


def mkdirs(dirpath):
    """equivelent to "mkdir -p"
    Make directory in dirpath, if its parent directories do not exist, create its parent directories too.
    """
    pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    global _seed
    _seed = int(seed)


def get_seed():
    global _seed
    return _seed
