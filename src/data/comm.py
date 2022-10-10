# Created on Mon Oct 10 2022 by Chuyang Zhao
from torch.utils import data


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
