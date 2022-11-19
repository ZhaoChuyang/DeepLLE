# Created on Mon Oct 10 2022 by Chuyang Zhao
from typing import List, Optional
import torch.utils.data as torchdata
import itertools
from .comm import ToIterableDataset, CommISPDataset
from .samplers import TrainingSampler, InferenceSampler
from .catalog import DATASET_CATALOG


__all__ = ['build_batch_data_loader', 'build_train_loader', 'build_test_loader']



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
    names
):
    """
    Args:
        names (str or list[str]): dataset name or a list of dataset names.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DATASET_CATALOG.get(name) for name in names]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset: {} is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
    return dataset_dicts


def build_train_loader(
    names = None,
    batch_size = 1,
    transforms = None,
    dataset = None,
    sampler=None,
    num_workers=0,
    collate_fn=None
):
    """
    Build a train loader.

    Args:
        names (str or list[str]): dataset name or a list of dataset names, you must provide at least one of dataset or names.
        batch_size (int): total batch size, the batch size each GPU got is batch_size // num_gpus.
        transforms (torchvision.Transforms): transforms applied to dataset.
        dataset (torchdata.Dataset or torchdata.IterableDataset): instantiated dataset, you must provide at least one of dataset or names.
        sampler (torchdata.sampler.Sampler): if you use map-style dataset, default is TrainingSampler. You should not provide this if you use iterable-style dataset.
        num_worker (int): num_worker of the dataloader.
        collate_fn (callable): use trivial_collate_fn by default.

    Note: Typically, if you want to use a dataset for a long time, you can
    register it in data/datasets, so that it can be easily loaded just
    given its registered name. But if you want to do some temporary
    experiments on a dataset, you can implement it as data.Dataset and
    manually load it.
    """
    if dataset is None:
        dataset_dicts = get_isp_dataset_dicts(names)
        dataset = CommISPDataset(dataset_dicts, True, transforms)
    
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"

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
