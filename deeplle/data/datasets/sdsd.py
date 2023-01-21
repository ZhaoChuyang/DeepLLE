# Created on Tue Jan 12 2023 by Chuyang Zhao
import os
from ...utils import check_path_is_image, check_path_exists
from ..catalog import DATASET_CATALOG


def load_sdsd_dataset(root: str, format: str, scene: str, **kwargs):
    """
    SDSD dataset is loaded as image dataset. The image frames will be
    concatenated to video sequence in `CommVideoISPDataset`.

    Args:
        root (str): path to the root directory of the SDSD dataset.
        format (str): data format, should be one of 'numpy' | 'image'.
        scene (str): scene name, should be one of 'indoor' | 'outdoor' | 'all'.
    """
    assert format in ['numpy', 'image'], 'format should be one of "numpy" | "image"'
    assert scene in ['indoor', 'outdoor', 'all'], 'scene should be one of "indoor" | "outdoor" | "all"'

    dataset_dicts = []
    scenes = ['indoor', 'outdoor'] if scene == 'all' else [scene]
    
    for scene in scenes:
        if format == 'numpy':
            dirname = "{}_np".format(scene)
        else:
            dirname = "{}".format(scene)
        
        gt_dir = os.path.join(root, dirname, "GT")
        lq_dir = os.path.join(root, dirname, "input")
        for dirpath, _, filenames in os.walk(gt_dir):
            for filename in filenames:
                if filename.endswith(".npy"):
                    tmp_path = dirpath[len(gt_dir):]
                    tmp_path = tmp_path[1:] if tmp_path.startswith("/") else tmp_path
                    lq_path = os.path.join(lq_dir, tmp_path, filename)
                    # some of the files are missing, we skip them
                    if not check_path_exists(lq_path): continue
                    gt_path = os.path.join(dirpath, filename)
                    data_dict = {
                        "input_path": lq_path,
                        "target_path": gt_path,
                        **kwargs
                    }
                    dataset_dicts.append(data_dict)
    
    return dataset_dicts


def register_sdsd_dataset(name: str, root: str, format: str, scene: str, **kwargs):
    """
    Register SDSD dataset with given name. The registered dataset
    can be used directly by specifying the registered name in
    the config.

    Args:
        name (str): name of the registered dataset.
        root (str): path to the root directory of the SDSD dataset.
        format (str): data format, should be one of 'numpy' | 'image'.
        scene (str): scene name, should be one of 'indoor' | 'outdoor' | 'all'.
    """
    DATASET_CATALOG.register(name, lambda: load_sdsd_dataset(root, format, scene, **kwargs))


if __name__ == '__main__':
    """
    Test SDSD Dataset, run with:
    $ python -m deeplle.data.datasets.sdsd
    """
    from IPython import embed
    sdsd_indoor = load_sdsd_dataset("/home/chuyang/Workspace/datasets/SDSD", "numpy", "indoor")
    sdsd_outdoor = load_sdsd_dataset("/home/chuyang/Workspace/datasets/SDSD", "numpy", "outdoor")
    sdsd_all = load_sdsd_dataset("/home/chuyang/Workspace/datasets/SDSD", "numpy", "all")
    embed()

    
        


