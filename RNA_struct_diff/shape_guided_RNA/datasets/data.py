import os
from os.path import join

import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

ROOT = os.path.dirname(os.path.abspath(__file__))

'''
class RNAFast(data.Dataset):
    def __init__(self, root=ROOT, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        if split not in ('train', 'val', 'test'):
            raise ValueError("split must be one of {'train', 'val', 'test'}")

        if not self._check_exists():
            raise RuntimeError(f'Dataset not found at {self.root}/preprocessed/{split}.npy')

        self.data = torch.from_numpy(np.load(join(self.root, 'preprocessed', f'{split}.npy')))

    def __getitem__(self, index):
        seq = self.data[index].long()

        if self.transform:
            seq = self.transform(seq)

        return seq

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        for split in ('train', 'val', 'test'):
            if not os.path.exists(join(self.root, 'preprocessed', f'{split}.npy')):
                return False
        return True


def collate_pad_batch(batch):
    """Pad RNA sequences in a batch to the same length."""
    lengths = [len(seq) for seq in batch]
    padded_seqs = pad_sequence(batch, batch_first=True, padding_value=0)  # 'A' = 0 assumed
    return padded_seqs, torch.tensor(lengths)


def get_data(args):
    if args.dataset == 'tRNA':
        root = "/home/georg/Documents/projects/discrete_diffusion/data/same_len"
    else:
        raise ValueError
    train_set = RNAFast(root=root, split='train')
    val_set = RNAFast(root=root, split='val')
    test_set = RNAFast(root=root, split='test')

    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=args.pin_memory,
                                              collate_fn=collate_pad_batch)

    valloader = torch.utils.data.DataLoader(val_set,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=args.pin_memory,
                                            collate_fn=collate_pad_batch)

    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=args.pin_memory,
                                             collate_fn=collate_pad_batch)

    # Attach metadata to args for convenience
    args.data_channels = 1
    args.variable_type = 'categorical'
    args.num_classes = 4  # A, C, G, U
    args.data_size = 'variable'  # sequences are variable-length
    #data_shape = None

    return trainloader, valloader, args.num_classes
import math
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def add_data_args(parser):
    parser.add_argument('--dataset', type=str, default='rna')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--rna_vocab', type=int, default=4)  # A,C,G,U,N

def get_data_id(args):
    return f"{args.dataset}"

def get_plot_transform(args):
    return lambda x: x  # Identity, no plotting transforms for RNA
'''
'''
def get_data(args):
    assert args.dataset == 'rna'

    # You define this: should return (tensor of token indices, optional label)
    train = RNADataset(split='train')
    test = RNADataset(split='test')

    # Pad batch to max length in each batch
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        lengths = [len(seq) for seq in sequences]
        padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)  # Assuming 0 is pad or 'A'
        labels = torch.tensor(labels) if labels[0] is not None else None
        return padded_seqs, lengths, labels

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory,
                              collate_fn=collate_fn)

    eval_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_memory,
                             collate_fn=collate_fn)

    data_shape = None  # Variable length
    num_classes = args.rna_vocab  # E.g., 5

    return train_loader, eval_loader, data_shape, num_classes

'''
import os
from os.path import join

import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

ROOT = os.path.dirname(os.path.abspath(__file__))


BUCKET_SIZES = [100, 200, 600]

class RNADataset(data.Dataset):
    def __init__(self, root, split='train', bucket_size=100):
        """
        Args:
            data_dir (str): directory containing *_seqs_len{bucket}.npy and *_maps_len{bucket}.npy
            split (str): 'train' or 'val'
            bucket_size (int): currently only 100 is supported
        """
        seq_path = os.path.join(root, f"{split}_seqs_len{bucket_size}.npy")
        map_path = os.path.join(root, f"{split}_maps_len{bucket_size}.npy")

        self.seqs = torch.from_numpy(np.load(seq_path)).long()      # (N, 100)
        self.maps = torch.from_numpy(np.load(map_path)).float()     # (N, 100, 100)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.maps[idx]

'''
def get_dataloader(data_dir, split='train', batch_size=32, shuffle=True, num_workers=0):
    """
    Returns DataLoader for bucket size 100 only.
    """
    dataset = RNADataset(root=root, split=split, bucket_size=100)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
'''


def get_data(args, root=ROOT):
    if args.dataset == "structRNA":
        root = "/home/georg/Documents/projects/discrete_diffusion/data/rna_structured"
        #data_root = os.path.abspath(os.path.join(ROOT, "../../data/same_len"))
        #print("[DEBUG] Using data root:", data_root)
    else:
        raise ValueError
    train_set = RNADataset(root=root, split='train')
    val_set = RNADataset(root=root, split='val')
    #test_set = RNAFast(root=root, split='test')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=args.pin_memory)



    args.data_channels = 1
    args.variable_type = 'categorical'
    args.num_classes = 4  # A, C, G, U, N (input vocab size)
    args.data_size = 'variable'

    return train_loader, val_loader, args.num_classes

def add_data_args(parser):
    parser.add_argument('--dataset', type=str, default='rna')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--rna_vocab', type=int, default=4)  # A,C,G,U,N

def get_data_id(args):
    return f"{args.dataset}"

def get_plot_transform(args):
    return lambda x: x  # Identity, no plotting transforms for RNA