# Created on Mon Oct 10 2022 by Chuyang Zhao
from typing import List, Optional, Union
import torch.utils.data as torchdata
import itertools
import logging
from deeplle.data.common import ToIterableDataset, CommISPDataset, CommVideoDataset
from deeplle.data.samplers import TrainingSampler, BalancedSampler, InferenceSampler
from deeplle.data.catalog import DATASET_CATALOG
from deeplle.utils.config import configurable, ConfigDict
from deeplle.data.transforms import build_image_transforms, build_video_transforms


__all__ = ['build_batch_data_loader', 'build_isp_train_loader', 'build_test_loader']


logger = logging.getLogger(__name__)


def build_batch_data_loader(
    dataset,
    sampler,
    batch_size,
    *,
    num_workers=0,
    collate_fn=None
):
    """
    Build a batched dataloader.
    """
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)
    
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn
    )


def get_isp_dataset_dicts(
    names: Union[str, List[str]],
):
    """
    Args:
        names (str or list[str]): dataset name or a list of dataset names.

    Returns:
        merged list of all dataset dicts.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DATASET_CATALOG.get(name) for name in names]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset: {} is empty!".format(dataset_name)

    # combine dataset dicts
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    return dataset_dicts


def get_isp_dataset_sizes(
    names
):
    """
    Get the sizes of all datasets specified by names.

    Args:
        names (str or list[str]): dataset name or a list of dataset names.

    Returns:
        list of the sizes of all datasets specified by names.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names

    dataset_sizes = [len(DATASET_CATALOG.get(name)) for name in names]

    return dataset_sizes


def _isp_train_loader_from_config(cfg: ConfigDict, type: str, *, dataset = None):
    """
    Args:
        cfg (ConfigDict): cfg is the config dict of the train data factory: `cfg.data_factory.train`.
        type (str): dataset type, one of 'image' | 'video'.
        dataset (optional): provide dataset if you don't want to construct dataset using dataset names
            specified in `cfg.names`.
    """
    if dataset is not None:
        names = None
    else:
        names = cfg.names

    assert type in ["video", "image"], f"Dataset type can only be 'image' or 'video'."
    video_kwargs = {}
    if type == "video":
        video_kwargs = {
            "num_frames": cfg.num_frames,
            "padding_mode": cfg.padding_mode,
        }
        
    if type == 'video':
        transforms = build_video_transforms(cfg.transforms)
    else:
        transforms = build_image_transforms(cfg.transforms)

    return {
        "names": names,
        "dataset": dataset,
        "idaug_datasets": cfg.idaug_datasets,
        "batch_size": cfg.batch_size,
        "transforms": transforms,
        "type": type,
        "sampler": cfg.sampler,
        "num_workers": cfg.num_workers,
        **video_kwargs,
    }


@configurable(from_config=_isp_train_loader_from_config)
def build_isp_train_loader(
    names=None,
    dataset=None,
    type="image",
    idaug_datasets=[],
    shuffle=True,
    batch_size=1,
    transforms=None,
    sampler=None,
    num_workers=0,
    collate_fn=None,
    # video dataset configs, used when type is 'video'
    num_frames=0,
    padding_mode="reflection",
):
    """
    Build a train loader.

    Args:
        names (str or list[str]): dataset name or a list of dataset names, you must provide at least one of dataset or names.
        dataset (torchdata.Dataset or torchdata.IterableDataset): instantiated dataset, you must provide at least one of dataset or names.
        type (str): dataset type, must be one of 'image' | 'video'.
        idaug_datasets (list): the names of the datasets you want to do identity augmentation.
        shuffle (bool): whether to shuffle the dataset. For image dataset, typically it is True, but for video dataset it is False.
        batch_size (int): total batch size, the batch size each GPU got is batch_size // num_gpus.
        transforms (torchvision.Transforms): transforms applied to dataset.
        sampler (str): specify this argument if you use map-style dataset, by default use the TrainingSampler. You should not provide this if you use iterable-style dataset.
        num_worker (int): num_worker of the dataloader.
        collate_fn (callable): use trivial_collate_fn by default.
        num_frames (int): if video_data is True, num_frames should be provided which is the number of frames in a single input video sequence.
        padding_mode (str): padding mode used for video dataset constrcution. Refer to `CommVideoISPDataset` for more details.

    Note: Typically, if you commomly use a dataset, you can
    register it in data/datasets, so that it can be easily loaded just
    by its registered name. But if you want to do some temporary
    experiments on a dataset, you can implement it as data.Dataset and
    explicitly provide it when calling this function.
    """
    if dataset is None:
        dataset_dicts = get_isp_dataset_dicts(names)
        dataset_sizes = get_isp_dataset_sizes(names)
        # TODO: show datasets information
        if type == "image":
            dataset = CommISPDataset(dataset_dicts, True, transforms, idaug_datasets)
        elif type == "video":
            dataset = CommVideoDataset(dataset_dicts, num_frames, padding_mode, True, transforms)
        else:
            logger.error(f"Dataset type must be 'image' or 'video', got {type}.")
            raise RuntimeError
    
    if type == 'video' and shuffle is True:
        logger.warning("Video dataset should not be shuffled, set shuffle to False.")
        shuffle = False

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler == "TrainingSampler":
            sampler = TrainingSampler(sum(dataset_sizes), shuffle=shuffle)
        elif sampler == "BalancedSampler":
            sampler = BalancedSampler(dataset_sizes, shuffle=shuffle)
        else:
            raise NotImplementedError(f"sampler: {sampler} is not implemented.")
    
    return build_batch_data_loader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


def build_test_loader(
    names = None,
    dataset = None,
    transforms = None,
    batch_size: int = 1,
    sampler=None,
    num_workers: int = 0,
    collate_fn = None,
):
    """
    Build a test loader.
    """
    if dataset is None:
        dataset_dicts = get_isp_dataset_dicts(names)
        dataset = CommISPDataset(dataset_dicts, False, transforms)
    else:
        assert transforms is None, "You should not provide transforms when you use dataset to construct the dataloader."
    
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn
    )


def trivial_batch_collator(batch):
    """
    A batch collator that do not do collation on the
    batched data but directly returns it as a list.
    """
    return batch
