from torch.utils import data


class BaseISPDataset(data.Dataset):
    def __init__(self, root, transforms: lambda x: x, mode: str = 'train'):
        assert mode in ['train', 'eval'], "Expecting dataset mode being 'train' or 'eval', got {} instead.".format(mode)
        self.root = root
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
