# Created on Fri Oct 14 2022 by Chuyang Zhao
import os
import random
from ...utils import check_path_is_image, find_files
import random
from ..catalog import DATASET_CATALOG


def load_sice_dataset(root: str, split: str, low_light: bool = True, seed: int = 0, idaug: bool = False):
    """
    Load the SICE dataset, introduced in "Learning a Deep Single Image
    Contrast Enhancer from Multi-Exposure Images".

    Args:
        root (str): Path to the root directory. The root directory
            should contains two directory "Dataset_Part1" and "Dataset_Part2".
        low_light (bool): If only use low light images to construct dataset. You must
            put the brightness label file "brightness_labels.txt" in the root directory.
        split (str): One of "train", "val", "test" and "all". According
            to the author paper, they split the dataset in to train, val and test
            set randomly with a ratio of 7:1:2. Here we following this setting.
        seed (int): Random seed to shuffle the dataset. By default is 0.
    
    Returns:
        a list of dataset dicts. Dataset dict contains:
            * image_path: path to the input image
            * target_path: path to the target image
    
    Note: this function does not read image and 'image' does not
    exist in the dict fields.

    """
    assert split in ['train', 'val', 'test', 'all'], "split can only be 'train', 'val', 'test' or 'all'. Got {}.".format(split)

    dataset = []

    brightness_labels = {}
    with open(os.path.join(root, "brightness_labels.txt")) as fb:
        for line in fb.readlines():
            filename, brightness = line.split(" ")
            brightness_labels[filename] = float(brightness)

    for imgdir in os.listdir(os.path.join(root, "Dataset_Part1")):
        if imgdir == "Label": continue
        for filename in os.listdir(os.path.join(root, "Dataset_Part1", imgdir)):
            image_path = os.path.join(root, "Dataset_Part1", imgdir, filename)
            
            relative_path = os.path.join("Dataset_Part1", imgdir, filename)
            brightness = brightness_labels[relative_path]
            if low_light and brightness > 0.5: continue

            if not check_path_is_image(image_path):
                continue
            target_paths = find_files(os.path.join(root, "Dataset_Part1", "Label", f"{imgdir}.*"))
            target_path = target_paths[0]
            dataset.append({"image_path": image_path, "target_path": target_path})
            if idaug:
                dataset.append({"image_path": target_path, "target_path": target_path})
    
    for imgdir in os.listdir(os.path.join(root, "Dataset_Part2")):
        if imgdir == "Label": continue
        for filename in os.listdir(os.path.join(root, "Dataset_Part2", imgdir)):
            image_path = os.path.join(root, "Dataset_Part2", imgdir, filename)

            relative_path = os.path.join("Dataset_Part2", imgdir, filename)
            brightness = brightness_labels[relative_path]
            if low_light and brightness > 0.5: continue

            if not check_path_is_image(image_path):
                continue
            target_paths = find_files(os.path.join(root, "Dataset_Part2", "Label", f"{imgdir}.*"))
            target_path = target_paths[0]
            dataset.append({"image_path": image_path, "target_path": target_path})
            if idaug:
                dataset.append({"image_path": target_path, "target_path": target_path})
    
    random.seed(seed)
    random.shuffle(dataset)

    size = len(dataset)
    val_start = int(size * 0.7)
    test_start = int(size * 0.8)

    if split == 'train':
        return dataset[:val_start]
    if split == 'val':
        return dataset[val_start:test_start]
    if split == 'test':
        return dataset[test_start:]
    if split == 'all':
        return dataset


def register_sice_dataset(name: str, root: str, split: str, idaug: bool):
    """
    Register the SICE dataset in DATASET_CATALOG.

    Args:
        name (str): register name.
        root, split: Refer to `load_sice_dataset` for details.
    """
    DATASET_CATALOG.register(name, lambda: load_sice_dataset(root, split, idaug))


if __name__ == '__main__':
    """
    Test load_sice_dataset.
    """
    root = '/data/chuyang/datasets/SICE'
    from IPython import embed
    train = load_sice_dataset(root, 'train')
    val = load_sice_dataset(root, 'val')
    test = load_sice_dataset(root, 'test')
    trainval = load_sice_dataset(root, 'trainval')
    all = load_sice_dataset(root, 'all')
    embed()