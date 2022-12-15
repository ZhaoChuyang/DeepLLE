# Created on Tue Nov 15 2022 by Chuyang Zhao
import os
from ...utils import check_path_is_image, check_path_exists
from ..catalog import DATASET_CATALOG


def load_ve_lol_dataset(root: str, category: str, split: str, identity_aug: bool = True):
    """
    Args:
        root (str): path to the root directory of VE-LOL. Refer to `deepisp/data/datasets/README.md` for the file structure inside root.
        categoty (str): synthesized or real captured data, should be in ["syn", "real", "all"].
        split (str): split of the dataset, should be in ["train", "test", "all"].
        identity_aug (bool): whether use the normal image as training pair, i.e. the input and target image are both the same normal image.

    Returns:
        a list of dataset dicts. Dataset dict contains:
            * image_path: path to the input image
            * target_path: path to the target image
    
    Note: this function does not read image and 'image' does not
    exist in the dict fields.
    """
    assert split in ["train", "test", "all"], 'split should be in ["train", "test", "all"]'
    assert category in ["syn", "real", "all"], 'category should be in ["syn", "real", "all"]'

    train_dicts = []
    test_dicts = []

    if category in ['syn', 'all']:
        # add train splits
        low_dir = os.path.join(root, "VE-LOL-L-Syn/VE-LOL-L-Syn-Low_train")
        high_dir = os.path.join(root, "VE-LOL-L-Syn/VE-LOL-L-Syn-Normal_train")
        assert check_path_exists(low_dir) and check_path_exists(high_dir)

        for low_path, high_path in _iter_subdir(low_dir, high_dir):
            record = {}
            record["image_path"] = low_path
            record["target_path"] = high_path
            train_dicts.append(record)

            if identity_aug:
                record = {}
                record["image_path"] = high_path
                record["target_path"] = high_path
                train_dicts.append(record)
        
        # add test splits
        low_dir = os.path.join(root, "VE-LOL-L-Syn/VE-LOL-L-Syn-Low_test")
        high_dir = os.path.join(root, "VE-LOL-L-Syn/VE-LOL-L-Syn-Normal_test")
        assert check_path_exists(low_dir) and check_path_exists(high_dir)

        for low_path, high_path in _iter_subdir(low_dir, high_dir):
            record = {}
            record["image_path"] = low_path
            record["target_path"] = high_path
            test_dicts.append(record)
    
    if category in ['real', 'all']:
        # add train splits
        low_dir = os.path.join(root, "VE-LOL-L-Cap-Full/VE-LOL-L-Cap-Low_train")
        high_dir = os.path.join(root, "VE-LOL-L-Cap-Full/VE-LOL-L-Cap-Normal_train")
        assert check_path_exists(low_dir) and check_path_exists(high_dir)

        for low_path, high_path in _iter_subdir(low_dir, high_dir):
            record = {}
            record["image_path"] = low_path
            record["target_path"] = high_path
            train_dicts.append(record)

            if identity_aug:
                record = {}
                record["image_path"] = high_path
                record["target_path"] = high_path
                train_dicts.append(record)

        # add test splits
        low_dir = os.path.join(root, "VE-LOL-L-Cap-Full//VE-LOL-L-Cap-Low_test")
        high_dir = os.path.join(root, "VE-LOL-L-Cap-Full/VE-LOL-L-Cap-Normal_test")
        assert check_path_exists(low_dir) and check_path_exists(high_dir)

        for low_path, high_path in _iter_subdir(low_dir, high_dir):
            record = {}
            record["image_path"] = low_path
            record["target_path"] = high_path
            test_dicts.append(record)
    
    if split == 'train':
        return train_dicts
    if split == 'test':
        return test_dicts
    if split == 'all':
        return train_dicts + test_dicts


def _iter_subdir(low_dir, high_dir):
    filenames = os.listdir(low_dir)
    for filename in filenames:
        if not check_path_is_image(filename): continue
        low_path = os.path.join(low_dir, filename)
        filename = filename.replace("low", "normal")
        high_path = os.path.join(high_dir, filename)
        yield low_path, high_path


def register_ve_lol_dataset(name: str, root: str, category: str, split: str):
    """
    Register lol dataset with given name. The rigistered dataset
    can be used directly by specifying the registered name in
    the config.

    Args:
        name (str): name of the registered dataset.
        root (str): root path of the dataset. Refer to load_lol_dataset() for more details.
        split (str): split type of the dataset. Refer to load_lol_dataset() for more details.
    
    """
    DATASET_CATALOG.register(name, lambda: load_ve_lol_dataset(root, category, split))



if __name__ == '__main__':
    """
    Test the VE-LOL dataset loader.
    """
    from IPython import embed
    root = "/data/chuyang/datasets/VE-LOL"
    syn_train = load_ve_lol_dataset(root, "syn", "train")
    syn_test = load_ve_lol_dataset(root, "syn", "test")
    real_train = load_ve_lol_dataset(root, "real", "train")
    real_test = load_ve_lol_dataset(root, "real", "test")
    all = load_ve_lol_dataset(root, "all", "all")
    embed()
    