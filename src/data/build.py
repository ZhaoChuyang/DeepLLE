# Created on Mon Oct 10 2022 by Chuyang Zhao
from .comm import ToIterableDataset, CommISPDataset
import torch.utils.data as torchdata
from .samplers import TrainingSampler, InferenceSampler
from .catalog import _DatasetCatalog


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


def build_train_loader(
    dataset,
    batch_size,
    transforms,
    sampler=None,
    num_workers=0,
    collate_fn=None
):
    """
    Build a train loader.

    Args:
        dataset: list or data.Dataset instance. 

    Typically, if you want to use a dataset for a long time, you can
    register it in data/datasets, so that it can be easily loaded just
    given its registered name. But if you want to do some temporary
    experiments on a dataset, you can implement it as data.Dataset and
    manually load it.
    """
    if isinstance(dataset, list):
        dataset = CommISPDataset(dataset, True, transforms)
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
    dataset,
    transforms,
    batch_size: int = 1,
    sampler=None,
    num_workers: int = 0,
    collate_fn = None,
):
    """
    Build a test loader.
    """
    if isinstance(dataset, list):
        dataset = CommISPDataset(dataset, False, transforms)
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
