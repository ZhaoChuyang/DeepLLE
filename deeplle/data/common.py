# Created on Mon Oct 10 2022 by Chuyang Zhao
import itertools
from typing import Dict, List
from torch.utils import data
from PIL import Image
from .data_utils import generate_frame_indices


def _shard_iterator_dataloader_worker(iterable):
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # do nothing
        yield from iterable
    else:
        yield from itertools.islice(iterable, worker_info.id, None, worker_info.num_workers)


class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """
    def __init__(self, dataset: data.Dataset, sampler: data.Sampler, shard_sampler: bool = True):
        """
        Args:
            dataset: an old-style dataset with ``__getitem__``
            sampler: a cheap iterable that produces indices to be applied on ``dataset``.
            shard_sampler: whether to shard the sampler based on the current pytorch data loader
                worker id. When an IterableDataset is forked by pytorch's DataLoader into multiple
                workers, it is responsible for sharding its data based on worker id so that workers
                don't produce identical data.

                Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
                and this argument should be set to True. But certain samplers may be already
                sharded, in that case this argument should be set to False.
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, data.Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler

    def __iter__(self):
        if self.shard_sampler:
            sampler = self.sampler
        else:
            sampler = _shard_iterator_dataloader_worker(self.sampler)
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
    1. Read the image from path.
    2. Apply transforms on the image.
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


class CommVideoISPDataset(data.Dataset):
    """
    Construct commom video ISP dataset from a list of dataset dicts.
    """
    def __init__(self, dataset_dicts: List[Dict], num_frames: int, padding_mode: str, is_train: bool, transforms):
        """
        Args:
            dataset_dicts (list): a list of dataset dicts. dict should contains 'image_path' and 'target_path'.
            num_frames (int): number of frames to be concatenated into a video sequence. It should be an odd number.
            padding (str): Padding mode, one of 'replicate' | 'reflection' | 'reflection_circle' | 'circle'
                Examples:
                >>> current_idx = 0, num_frames = 5
                The generated frame indices under different padding mode:
                    replicate: [0, 0, 0, 1, 2]
                    reflection: [2, 1, 0, 1, 2]
                    reflection_circle: [4, 3, 0, 1, 2]
                    circle: [3, 4, 0, 1, 2]
            is_train (bool): set to True if you want to use this dataset in training mode, otherwise for inference mode.
                In the training mode input and target image pairs is returned, otherwise only input image is returned.
            transforms (callable): do transforms on image pairs or single image.
        
        TODO: the images can be cached for fast loading.
        """
        self.dataset_dicts = dataset_dicts
        self.gt_paths = [record["target_path"] for record in dataset_dicts] # paths of the ground truth images
        self.lq_paths = [record["image_path"] for record in dataset_dicts] # paths of the low quality images
        self.total_frames = len(self.lq_paths)
        assert num_frames % 2 == 1, "num_frames should be odd"
        self.num_frames = num_frames
        assert padding_mode in ["replicate", "reflection", "reflection_circle", "circle"], "padding mode should be one of [replicate, reflection, reflection_circle, circle]"
        self.padding_mode = padding_mode
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # if self.is_train:
        lq_indices = generate_frame_indices(idx, self.total_frames, self.num_frames, padding=self.padding_mode)
        lq_img_paths = [self.lq_paths[i] for i in lq_indices]
        lq_images = [Image.open(path) for path in lq_img_paths]
        if self.is_train:
            gt_img_path = self.gt_paths[idx]
            gt_image = Image.open(gt_img_path)
        else:
            gt_image = None
        # do transforms on the low quality images (list of n image frames) and ground truth image (single image).
        # after transforming, input should be a tensor of shape (n, c, h, w), target should be a tensor of shape (c, h, w)
        input, target = self.transforms(lq_images, gt_image)

        record = self.dataset_dicts[idx]
        record.update({
            "input": input, # tensor: (n, c, h, w)
            "target": target, # tensor: (c, h, w) if target is available else None
        })
        return record

        

        
