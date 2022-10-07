import os
import json
import pathlib



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
