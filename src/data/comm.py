# Created on Mon Oct 10 2022 by Chuyang Zhao
from typing import Dict, List
from torch.utils import data
from PIL import Image



class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """
    def __init__(self, dataset: data.Dataset, sampler: data.Sampler):
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, data.Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        for idx in self.sampler:
            yield self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


class CommISPDataset(data.Dataset):
    """
    Construct common ISP dataset from a list of dataset dicts. You can also
    construct your own dataset not based on the list of dataset dicts,
    by this way you need to override the load dataset method in the Trainer.

    The CommISPDataset does the following:
    1. Read the image from path
    2. Apply transforms on the image
    3. TODO: may be save the image in shared RAM so that different workers
    can load the same one without copying.

    Args:
        datasets (list): a list of dataset dicts. dict should contains 'image_path'
            and 'target_path' if is_train is True, otherwise it should contains 'image_path'.
        is_train (bool): set True if you want to use this dataset in training mode, in which
            both input image and its target pair are assumed to exist. If you set it to False,
            we assume you load the dataset in testing mode, in which only input images exist.
        transforms: do transforms to image pairs or single image only.

    """
    def __init__(self, datasets: List[Dict], is_train: bool, transforms):
        self.datasets = datasets
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, idx):
        """
        Returns:
            if self.is_train is True, the returned dict contains:
                * image_path: path to the input image
                * target_path: path to the target image
                * image: input image in tensor
                * target: target image in tensor
                * other keys in the dict
            if self.is_train is False, the returned dict contains:
                * image_path: path to the input image
                * image: input image in tensor
                * other keys in the dict
        """
        if self.is_train:
            record = self.datasets[idx]
            image = Image.open(record['image_path'])
            target = Image.open(record['target_path'])
            image, target = self.transforms(image, target)
            record['image'] = image
            record['target'] = target
            return record
        else:
            record = self.datasets[idx]
            
            if 'target_path' in record:
                image = Image.open(record['image_path'])
                target = Image.open(record['target_path'])
                image, target = self.transforms(image, target)
                record['image'] = image
                record['target'] = target
            else:
                image = Image.open(record['image_path'])
                image = self.transforms(image)
                record['image'] = image
            
            return record
