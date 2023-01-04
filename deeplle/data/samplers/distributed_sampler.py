# Created on Wed Jan 04 2023 by Chuyang Zhao
from typing import Optional, List
import torch
from torch.utils.data.sampler import Sampler
import itertools

from deeplle.utils import comm


__all__ = ["TrainingSampler", "InferenceSampler", "BalancedSampler"]


class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)

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
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)
    
    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size)


class BalancedSampler(Sampler):
    """
    If you use the map-style dataset and your final dataset is consisted of datasets of different size,
    you can use the BalancingSampler to generate infinite data stream with each dataset is sampled the
    same number of times.
    """
    def __init__(self, dataset_sizes: List[int], shuffle: bool = True, seed: Optional[int] = None, buffer_size: int = 10000):
        self._shuffle = shuffle

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        self.dataset_sizes = dataset_sizes
        self.buffer_size = buffer_size
    
    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)

        counters = [0] * len(self.dataset_sizes)
        while True:
            indices = []
            base_num = 0
            for idx, dataset_size in enumerate(self.dataset_sizes):
                times = self.buffer_size // dataset_size
                remain = self.buffer_size % dataset_size
                indices.extend(times * [i % dataset_size + base_num for i in range(base_num + counters[idx], base_num + counters[idx] + dataset_size)])
                indices.extend([i % dataset_size + base_num for i in range(base_num + counters[idx], base_num + counters[idx] + remain)])
                counters[idx] = counters[idx] + remain
                base_num += dataset_size

            if self._shuffle:
                indices = torch.tensor(indices, dtype=torch.int64)
                yield from indices[torch.randperm(indices.size(0), generator=g)].tolist()
            else:
                yield from indices

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

                
class InferenceSampler(Sampler):
    """
    Sampler for inference dataset.
    Runs exactly one time for all samples in the test dataset.
    The stream the inference sampler created is not infinite.
    """
    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert self._size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
