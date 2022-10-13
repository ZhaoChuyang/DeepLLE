import os
from torch.utils import data
from PIL import Image
from .base import BaseISPDataset
from ...utils import check_path_is_image


class LOL(BaseISPDataset):
    def __init__(self, root, transforms = None, mode: str = 'train'):
        super().__init__(root, transforms, mode)

        self.train_root = os.path.join(root, 'our485')
        self.eval_root = os.path.join(root, 'eval15')
        
        self.train = []
        self.eval = []

        for filename in os.listdir(os.path.join(root, 'our485/high')):
            if not check_path_is_image(filename):
                continue
            src_path = os.path.join(root, 'our485/high', filename)
            tgt_path = os.path.join(root, 'our485/low', filename)
            self.train.append((src_path, tgt_path))
        
        for filename in os.listdir(os.path.join(root, 'eval15/high')):
            if not check_path_is_image(filename):
                continue
            src_path = os.path.join(root, 'eval15/high', filename)
            tgt_path = os.path.join(root, 'eval15/low', filename)
            self.eval.append((src_path, tgt_path))

    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        else:
            return len(self.eval)

    def __getitem__(self, idx):
        if self.mode == 'train':
            src_path, tgt_path = self.train[idx]
            src = Image.open(src_path)
            tgt = Image.open(tgt_path)
            
            src, tgt = self.transforms(src, tgt)

            return {
                "image": src,
                "target": tgt
            }
        else:
            src_path, tgt_path = self.eval[idx]
            src = Image.open(src_path)
            tgt = Image.open(tgt_path)
            
            src = self.transforms(src)
            tgt = self.transforms(tgt)

            return {
                "image": src,
                "target": tgt
            }
