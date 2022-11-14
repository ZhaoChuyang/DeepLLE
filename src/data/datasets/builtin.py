# Created on Fri Oct 14 2022 by Chuyang Zhao
import os
"""
Hard-code registering the builtin datasets, so you can directly
use these builtin datasets in config by its name.

All builtin datasets are assumed to be store in 'DeepISP/datasets'.
If you want to change the root directoy of datasets, set the system
enviroment "ISP_DATASETS" by:
    export ISP_DATASETS='~/path/to/directory'

Don't modify this file to change the root directory of datasets.

This file is intended to register builtin datasets only, please
don't register your custom datasets in this file.
"""
from .lol import register_lol_dataset
from .sice import register_sice_dataset
from .mbllen import register_mbllen_dataset


def register_all_lol(root):
    SPLITS = [
        ("lol_train", "LOL", "train"),
        ("lol_val", "LOL", "val"),
        ("lol_all", "LOL", "all")
    ]
    for name, dirname, split in SPLITS:
        register_lol_dataset(name, os.path.join(root, dirname), split)


def register_all_sice(root):
    SPLITS = [
        ("sice_train", "SICE", "train"),
        ("sice_val", "SICE", "val"),
        ("sice_trainval", "SICE", "trainval"),
        ("sice_all", "SICE", "all")
    ]
    for name, dirname, split in SPLITS:
        register_sice_dataset(name, os.path.join(root, dirname), split)


def register_all_mbllen(root):
    ALL = [
        ("mbllen_dark", "MBLLEN", False),
        ("mbllen_noisy", "MBLLEN", True),
    ]
    for name, dirname, noise in ALL:
        register_mbllen_dataset(name, os.path.join(root, dirname), noise)


_root = os.path.expanduser(os.getenv("ISP_DATASETS", "datasets"))
register_all_lol(_root)
register_all_sice(_root)
register_all_mbllen(_root)
