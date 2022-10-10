# Created on Mon Oct 10 2022 by Chuyang Zhao
from typing import Optional
import torch
from torch.utils.data.sampler import Sampler
from ... import utils


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices.

    For any map-style dataset, we need to convert it to IterableDataset using
    this sampler in order to get infinite stream.
    """
    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): Size of the dataset.
            shuffle (bool): Shuffle the dataset if set true.
            seed (int): Optional, the initial seed of the sampler. If None, will
                use the seed returned by utils.get_seed().
        """
        if not isinstance(size, int):
            raise TypeError("size of the TrainSampler expected to be an int. Got type {}".format(type(size)))
        if size <= 0:
            raise ValueError("size of the TrainSampler expected a positive int. Got {}".format(size))
        
        self._size = size
        self._shuffle = shuffle

        if seed is None:
            self._seed = utils.get_seed()

    def __iter__(self):
        yield from self._infinite_indices()
        
    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size)


class InferenceSampler(Sampler):
    """
    Sampler for inference dataset.
    Runs exactly one time for all samples in the test dataset.
    The stream the inference sampler created is not infinite.
    """
    def __init__(self, size: int):
        self._size = size
        self._indices = torch.arange(self._size).tolist()

    def __iter__(self):
        yield from self._indices

    def __len__(self):
        return self._size
