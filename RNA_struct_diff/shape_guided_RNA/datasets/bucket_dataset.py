import os
import pickle
import random
from collections import defaultdict
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import collections
PAD_TOKEN = 4
BUCKET_SIZES = [64, 128, 256, 640]
import sys

RNA_SS_data = collections.namedtuple('RNA_SS_data', 'data_fcn_2 seq_raw length name contact')
sys.modules['__main__'].RNA_SS_data = RNA_SS_data

def encode_sequence(seq):
    vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    return [vocab.get(base, PAD_TOKEN) for base in seq]


def create_contact_map(contacts, padded_length, seq_length=None):
    contact_map = np.zeros((padded_length, padded_length), dtype=np.float32)
    for i, j in contacts:
        if (seq_length is None and i < padded_length and j < padded_length) or \
           (seq_length is not None and i < seq_length and j < seq_length):
            contact_map[i, j] = 1
            contact_map[j, i] = 1
    return contact_map


def pad_array(arr, target_length, pad_value=PAD_TOKEN):
    padded = np.full((target_length,), pad_value, dtype=np.int64)
    padded[:len(arr)] = arr
    return padded

#helper class
from torch.utils.data import Dataset

class TupleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)


class BucketedRNADataset:
    def __init__(self, filepaths, bucket_sizes=BUCKET_SIZES, upsample=None):
        """
        upsample: dict mapping bucket_size -> replication factor
                  defaults to 1 if not given
        Example: {256: 2, 640: 4}
        """
        self.bucket_sizes = bucket_sizes

        if upsample is None:
            self.upsample = defaultdict(lambda: 1)   # all 1 by default
        else:
            self.upsample = defaultdict(lambda: 1, upsample)

        self.bucketed_data = defaultdict(list)
        for path in filepaths:
            with open(path, 'rb') as f:
                entries = pickle.load(f)
            for entry in entries:
                seq = encode_sequence(entry.seq_raw)
                L = len(seq)
                for b in self.bucket_sizes:
                    if L <= b:
                        padded_seq = pad_array(np.array(seq, dtype=np.int64), b)
                        padded_contact = create_contact_map(entry.contact, padded_length=b, seq_length=L)

                        sample = (torch.tensor(padded_seq), torch.tensor(padded_contact))
                        for _ in range(int(self.upsample[b])):  # replicate k times
                            self.bucketed_data[b].append(sample)
                        #self.bucketed_data[b].append((torch.tensor(padded_seq), torch.tensor(padded_contact)))
                        break
    def get_bucket_dataloaders(self, base_batch_size, shuffle, num_workers, pin_memory,ddp=False):
        def collate_fn(batch):
            seqs, maps = zip(*batch)
            return torch.stack(seqs), torch.stack(maps)


        min_bucket_size = min(self.bucketed_data.keys())
        dataloaders = {}
        for b, bucket in self.bucketed_data.items():
            dataset = TupleDataset(bucket)
            batch_size = max(1, base_batch_size * (min_bucket_size ** 2) // (b ** 2))
            sampler = None
            if ddp:
                sampler = DistributedSampler(dataset, shuffle=shuffle)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=shuffle if sampler is None else False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )
            dataloaders[b] = dataloader
        return dataloaders
        '''
        return {
            b: DataLoader(bucket,
                          batch_size=max(1, base_batch_size * (min_bucket_size ** 2) // (b ** 2)),
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          collate_fn=collate_fn)
            for b, bucket in self.bucketed_data.items()
        }
        '''

class MultiBucketLoader:
    def __init__(self, loaders_dict):
        self.loaders_dict = loaders_dict
        self.epoch = 0
        self.base_seed = 42

    def buckets(self):
        return list(self.loaders_dict.keys())

    def __iter__(self):
        bucket_keys = list(self.loaders_dict.keys())
        rng = random.Random(self.base_seed + self.epoch)
        rng.shuffle(bucket_keys)
        for key in bucket_keys:
            for batch in self.loaders_dict[key]:
                yield batch

    def __len__(self):
        return sum(len(loader) for loader in self.loaders_dict.values())

    def set_epoch(self, epoch):
        self.epoch = epoch
        for loader in self.loaders_dict.values():
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
    @property
    def dataset(self):
        # Flatten all datasets from inner loaders
        class CombinedDataset:
            def __len__(self_inner):  # define __len__ for compatibility
                return sum(len(loader.dataset) for loader in self.loaders_dict.values())

        return CombinedDataset()

def get_data(args, subset_fraction=1.0,upsample={256:2,640:2}):
    # Determine dataset file paths



    path_base = "/home/gback/discrete_diffusion/data/RNADiffFold_data/"
    if not hasattr(args, 'bin_splits'):
        args.bin_splits = BUCKET_SIZES
    if args.dataset == 'bpRNA':
        args.train_paths = [path_base+'bpRNA/TR0.cPickle']
        args.val_paths = [path_base+'bpRNA/VL0.cPickle']
    elif args.dataset == 'RNAStrAlign':
        args.train_paths = [path_base+'RNAStrAlign/train.cPickle']
        args.val_paths = [path_base+'RNADiffFold_data/RN.cPickle']
    elif args.dataset == 'all':
        args.train_paths = [
            path_base+'bpRNA/TR0.cPickle',
            path_base+'RNAStrAlign/train.cPickle'
        ]
        args.val_paths = [
            path_base+'bpRNA/VL0.cPickle',
            path_base+'RNAStrAlign/val.cPickle'
        ]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    try:
        import torch.distributed as dist
        ddp = dist.is_available() and dist.is_initialized()
    except ImportError:
        ddp = False
    train_dataset = BucketedRNADataset(args.train_paths, bucket_sizes=args.bin_splits,upsample=upsample)

    if subset_fraction < 1.0:
        for b in train_dataset.bucketed_data:
            data = train_dataset.bucketed_data[b]
            k = max(1, int(len(data) * subset_fraction))
            train_dataset.bucketed_data[b] = random.sample(data, k)

    val_dataset = BucketedRNADataset(args.val_paths, bucket_sizes=args.bin_splits)
    train_loaders = train_dataset.get_bucket_dataloaders(
        base_batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        ddp=ddp
    )

    val_loaders = val_dataset.get_bucket_dataloaders(
        base_batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        ddp=ddp
    )
    if not hasattr(args, 'num_classes'):
        args.num_classes = 4
    # Combine loaders into a single object that tracks buckets

    return MultiBucketLoader(train_loaders), MultiBucketLoader(val_loaders), args.num_classes


def add_data_args(parser):
    parser.add_argument('--dataset', type=str, default='rna')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--rna_vocab', type=int, default=4)  # A,C,G,U,N

def get_data_id(args):
    return f"{args.dataset}"
