import os
import torch
import torch.utils.data as data
import numpy as np


class RNADataset(data.Dataset):
    def __init__(self, root, split='train', vocab_size=4, max_len=None, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.vocab_size = vocab_size
        self.transform = transform

        assert split in ('train', 'val', 'test')
        path = os.path.join(self.root, f'{split}.npy')

        if not os.path.exists(path):
            raise RuntimeError(f"RNA data file not found: {path}")

        self.data = np.load(path, allow_pickle=True)  # List of 1D arrays
        self.max_len = max_len  # optional

    def __getitem__(self, index):
        seq = torch.tensor(self.data[index], dtype=torch.long)  # [L]

        if self.max_len is not None:
            seq = seq[:self.max_len]

        if self.transform is not None:
            seq = self.transform(seq)

        return seq  # You can return (seq, label) if needed

    def __len__(self):
        return len(self.data)