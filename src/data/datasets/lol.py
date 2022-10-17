# Created on Fri Oct 14 2022 by Chuyang Zhao
import os
from ...utils import check_path_is_image
from ..catalog import DATASET_CATALOG


def load_lol_dataset(root: str, split: str):
    """
    Load the LOL dataset into a list of dataset dicts.

    Args:
        root (str): path to the root directory of the lol dataset.
            which should contains two directories eval15 and our485.
        split (str): split type of the dataset, should be one of "train", "val", "all".
    
    Returns:
        a list of dataset dicts. Dataset dict contains:
            * image_path: path to the input image
            * target_path: path to the target image
    
    Note: this function does not read image and 'image' does not
    exist in the dict fields.
    """
    assert split in ['train', 'val', 'all'], "split type can only be one of 'train', 'val' and 'all'. Got: {}.".format(split)
    
    train_dicts = []
    val_dicts = []

    for filename in os.listdir(os.path.join(root, 'our485/high')):
        if not check_path_is_image(filename):
            continue
        record = {}
        src_path = os.path.join(root, 'our485/low', filename)
        tgt_path = os.path.join(root, 'our485/high', filename)
        record["image_path"] = src_path
        record["target_path"] = tgt_path
        train_dicts.append(record)
    
    for filename in os.listdir(os.path.join(root, 'eval15/high')):
        if not check_path_is_image(filename):
            continue
        src_path = os.path.join(root, 'eval15/low', filename)
        tgt_path = os.path.join(root, 'eval15/high', filename)
        record["image_path"] = src_path
        record["target_path"] = tgt_path
        val_dicts.append(record)\
    
    if split == 'train':
        return train_dicts
    if split == 'val':
        return val_dicts
    if split == 'all':
        return train_dicts + val_dicts


def register_lol_dataset(name: str, root: str, split: str):
    """
    Register lol dataset with given name. The rigistered dataset
    can be used directly by specifying the registered name in
    the config.

    Args:
        name (str): name of the registered dataset.
        root (str): root path of the dataset. Refer to load_lol_dataset() for more details.
        split (str): split type of the dataset. Refer to load_lol_dataset() for more details.
    
    """
    DATASET_CATALOG.register(name, lambda: load_lol_dataset(root, split))


if __name__ == '__main__':
    """
    Test the LOL dataset loader.
    """
    from IPython import embed
    root = "/data/chuyang/datasets/LOL"
    train = load_lol_dataset(root, "train")
    val = load_lol_dataset(root, "val")
    all = load_lol_dataset(root, "all")
    embed()
    