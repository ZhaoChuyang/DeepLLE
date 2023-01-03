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
        ("lol_train", "LOL", "train", False),
        ("lol_train_idaug", "LOL", "train", True),
        ("lol_val", "LOL", "val", False),
        ("lol_all", "LOL", "all", False),
        ("lol_all_idaug", "LOL", "all", True)
    ]
    for name, dirname, split, idaug in SPLITS:
        register_lol_dataset(name, os.path.join(root, dirname), split, idaug)


def register_all_sice(root):
    SPLITS = [
        ("sice_train", "SICE", "train", False),
        ("sice_train_idaug", "SICE", "train", True),
        ("sice_val", "SICE", "val", False),
        ("sice_test", "SICE", "test", False),
        ("sice_all", "SICE", "all", False),
        ("sice_all_idaug", "SICE", "all", True),
    ]
    for name, dirname, split, idaug in SPLITS:
        register_sice_dataset(name, os.path.join(root, dirname), split, idaug)


def register_all_mbllen(root):
    ALL = [
        ("mbllen_dark", "MBLLEN", False),
        ("mbllen_noisy", "MBLLEN", True),
    ]
    for name, dirname, noise in ALL:
        register_mbllen_dataset(name, os.path.join(root, dirname), noise)


def register_all_ve_lol(root):
    ALL = [
        ("ve_lol_syn_train", "VE-LOL", "syn", "train", False),
        ("ve_lol_syn_train_idaug", "VE-LOL", "syn", "train", True),
        ("ve_lol_syn_test", "VE-LOL", "syn", "test", False),
        ("ve_lol_real_train", "VE-LOL", "real", "train", False),
        ("ve_lol_real_train_idaug", "VE-LOL", "real", "train", True),
        ("ve_lol_real_test", "VE-LOL", "real", "test", False),
        ("ve_lol_all", "VE-LOL", "all", "all", False),
        ("ve_lol_all_idaug", "VE-LOL", "all", "all", True),
    ]
    for name, dirname, category, split, idaug in ALL:
        register_ve_lol_dataset(name, os.path.join(root, dirname), category, split, idaug)


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
        register_fivek_dataset(name, os.path.join(root, dirname), expert)


_root = os.path.expanduser(os.getenv("ISP_DATASETS", "datasets"))
register_all_lol(_root)
register_all_sice(_root)
register_all_mbllen(_root)
register_all_ve_lol(_root)
register_all_fivek(_root)
