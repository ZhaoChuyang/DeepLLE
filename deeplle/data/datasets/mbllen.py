import os
from ...utils import check_path_is_image
from ..catalog import DATASET_CATALOG


def load_mbllen_dataset(root: str, noise: bool, **kwargs):
    """
    Load MBLLEN dataset.

    Args:
        root (str): root dir of MBLLEN dataset, which should contains "train", "train_dark" and "train_lowlight" three sub-directories.
        noise (bool): set True to use low-light dataset with random poisson noise, otherwise use pure low_light dataset.
    
    Returns:
        a list of dataset dicts. Dataset dict contains:
            * image_path: path to the input image
            * target_path: path to the target image
    """
    train_dicts = []
    for filename in os.listdir(os.path.join(root, "train")):
        if check_path_is_image(filename):
            target_path = os.path.join(root, 'train', filename)
            if noise:
                image_path = os.path.join(root, 'train_lowlight', filename)
            else:
                image_path = os.path.join(root, 'train_dark', filename)
            
            data_dict = {
                "image_path": image_path,
                "target_path": target_path
            }
            data_dict.update(kwargs)
            train_dicts.append(data_dict)
    
    return train_dicts


def register_mbllen_dataset(name: str, root: str, noise: bool, **kwargs):
    """
    Register mbllen dataset with given name. The rigistered dataset
    can be used directly by specifying the registered name in
    the config.

    Args:
        name (str): name of the registered dataset.
        root (str): root path of the dataset. Refer to load_mbllen_dataset() for more details.
        noise (bool): type of the dataset. Refer to load_mbllen_dataset() for more details.
    
    """
    DATASET_CATALOG.register(name, lambda: load_mbllen_dataset(root, noise, **kwargs))



if __name__ == '__main__':
    from IPython import embed
    mbllen_dark = load_mbllen_dataset("/data/chuyang/datasets/MBLLEN", False)
    mbllen_noisy = load_mbllen_dataset("/data/chuyang/datasets/MBLLEN", True)
    embed()
