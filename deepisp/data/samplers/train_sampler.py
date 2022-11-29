# Created on Mon Oct 10 2022 by Chuyang Zhao
from typing import Optional, List
import torch
from torch.utils.data.sampler import Sampler
import copy
import random
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


class BalancedSampler(Sampler):
    """
    If you use the map-style dataset and your final dataset is consisted of datasets of different size,
    you can use the BalancingSampler to generate infinite data stream with each dataset is sampled the
    same number of times.
    """
    def __init__(self, dataset_sizes: List[int], shuffle: bool = True, seed: Optional[int] = None, buffer_size: int = 10000):
        self._shuffle = shuffle

        if seed is None:
            self._seed = utils.get_seed()

        self.dataset_sizes = dataset_sizes
        self.buffer_size = buffer_size
    
    def _infinite_indices(self):
        counters = [0] * len(self.dataset_sizes)
        while True:
            indices = []
            base_num = 0
            for idx, dataset_size in enumerate(self.dataset_sizes):
                times = self.buffer_size // dataset_size
                remain = self.buffer_size % dataset_size
                indices.extend(times * [i % dataset_size for i in range(base_num + counters[idx], base_num + counters[idx] + dataset_size)])
                indices.extend([i % dataset_size for i in range(base_num + counters[idx], base_num + counters[idx] + remain)])
                counters[idx] = counters[idx] + remain
                base_num += dataset_size
            
            if self._shuffle:
                random.shuffle(indices)

            yield from indices

    def _get_dataset_mult_factors(self, nums: List[int]):
        """
        use of `_get_dataset_mult_factors` is deprecated, because the generated indices is too large to load in memory.
        """
        if len(nums) == 1: return [nums[0]]

        # calculate the greatest common factor of nums
        _nums = copy.copy(nums)

        while len(_nums) >= 2:
            a = _nums.pop()
            b = _nums.pop()

            factor = self._gcd(a, b)
            _nums.append(factor)
        
        factor = _nums[0]

        # buffer_size = factor
        factors = []
        for n in nums:
            factors.append(n // factor)
            # buffer_size *=  (n // factor)
        # buffer_size *= len(nums)
        
        return factors

    def _gcd(self, a: int, b: int) -> int:
        if b == 0: return a
        return self._gcd(b, a % b)
    
    def __iter__(self):
        yield from self._infinite_indices()

                
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
