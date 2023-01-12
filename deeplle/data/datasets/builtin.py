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
from .ve_lol import register_ve_lol_dataset
from .fivek import register_fivek_dataset


def register_all_lol(root):
    SPLITS = [
        ("lol_train", "LOL", "train"),
        ("lol_val", "LOL", "val"),
        ("lol_all", "LOL", "all"),
    ]
    for name, dirname, split in SPLITS:
        register_lol_dataset(name, os.path.join(root, dirname), split, dataset_name=name)


def register_all_sice(root):
    SPLITS = [
        ("sice_train", "SICE", "train"),
        ("sice_val", "SICE", "val"),
        ("sice_test", "SICE", "test"),
        ("sice_all", "SICE", "all"),
    ]
    for name, dirname, split in SPLITS:
        register_sice_dataset(name, os.path.join(root, dirname), split, dataset_name=name)


def register_all_mbllen(root):
    ALL = [
        ("mbllen_dark", "MBLLEN", False),
        ("mbllen_noisy", "MBLLEN", True),
    ]
    for name, dirname, noise in ALL:
        register_mbllen_dataset(name, os.path.join(root, dirname), noise, dataset_name=name)


def register_all_ve_lol(root):
    ALL = [
        ("ve_lol_syn_train", "VE-LOL", "syn", "train"),
        ("ve_lol_syn_test", "VE-LOL", "syn", "test"),
        ("ve_lol_real_train", "VE-LOL", "real", "train"),
        ("ve_lol_real_test", "VE-LOL", "real", "test"),
        ("ve_lol_all", "VE-LOL", "all", "all"),
    ]
    for name, dirname, category, split in ALL:
        register_ve_lol_dataset(name, os.path.join(root, dirname), category, split, dataset_name=name)


def register_all_fivek(root):
    ALL = [
        ("fivek_a", "fivek", "a"),
        ("fivek_b", "fivek", "b"),
        ("fivek_c", "fivek", "c"),
        ("fivek_d", "fivek", "d"),
        ("fivek_e", "fivek", "e"),
        ("fivek_all", "fivek", "all"),
    ]
    for name, dirname, expert in ALL:
        register_fivek_dataset(name, os.path.join(root, dirname), expert, dataset_name=name)


_root = os.path.expanduser(os.getenv("ISP_DATASETS", "datasets"))
register_all_lol(_root)
register_all_sice(_root)
register_all_mbllen(_root)
register_all_ve_lol(_root)
register_all_fivek(_root)
