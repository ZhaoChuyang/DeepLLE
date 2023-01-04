import os
from ...utils import check_path_is_image
from ..catalog import DATASET_CATALOG


def load_fivek_dataset(root: str, expert: str):
    """
    Load the fivek dataset into a list of dataset dicts.

    Args:
        root (str): path to the root directory, which should contains six subdirectories:
            "a", "b", "c", "d", "e" and "raw".
        expert (str): annotation expert, should be one of "a", "b", "c", "d", "e" and "all".
    
    Returns:
        a list of dataset dicts. Dataset dict contains:
            * image_path: path to the input image
            * target_path: path to the target image
    
    Note: this function does not read image and 'image' does not
    exist in the dict fields.
    """
    assert expert in ["a", "b", "c", "d", "e", "all"], "expert should be chosen from ['a', 'b', 'c', 'd', 'e', 'all']"

    data_dicts = []

    if expert == "all":
        target_dir = [os.path.join(root, dir) for dir in ["a", "b", "c", "d", "e"]]
    else:
        target_dir = [os.path.join(root, expert)]
    
    input_dir = os.path.join(root, "raw")

    for filename in os.listdir(input_dir):
        if not check_path_is_image(filename):
            continue
        src_path = os.path.join(input_dir, filename)
        
        for tgt_dir in target_dir:
            tgt_path = os.path.join(tgt_dir, filename)
            data_dicts.append({
                "image_path": src_path,
                "target_path": tgt_path
            })
    
    return data_dicts


def register_fivek_dataset(name: str, root: str, expert: str):
    """
    Register the fivek dataset.
    """
    DATASET_CATALOG.register(
        name,
        lambda: load_fivek_dataset(root, expert)
    )


if __name__ == "__main__":
    """
    test the fivek dataset
    """
    fivek_all = load_fivek_dataset("/home/chuyang/Workspace/datasets/fivek", "all")
    fivek_a = load_fivek_dataset("/home/chuyang/Workspace/datasets/fivek", "a")
    from IPython import embed
    embed()
