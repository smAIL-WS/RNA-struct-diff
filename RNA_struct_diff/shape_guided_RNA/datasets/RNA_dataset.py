import torch
from torch.utils.data import Dataset, DataLoader

class TRNADataset(Dataset):
    def __init__(self, sequences, max_len=128, pad_val=-1):
        self.sequences = sequences
        self.token_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.max_len = max_len
        self.pad_val = pad_val

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokenized = [self.token_map.get(nt, 0) for nt in seq]
        tokenized = tokenized[:self.max_len]
        padding = [self.pad_val] * (self.max_len - len(tokenized))
        return torch.tensor(tokenized + padding, dtype=torch.long)

def trna_collate_fn(batch, pad_val=-1):
    batch = torch.stack(batch)
    mask = (batch != pad_val)
    batch[~mask] = 0  # Replace pad_val with zero for indexing
    return batch, mask

def load_rna_dataset(data_dir, batch_size, pad_idx=4, **kwargs):
    train_set = RNADataset(data_dir, split='train')
    val_set = RNADataset(data_dir, split='val')
    test_set = RNADataset(data_dir, split='test')

    collate = lambda batch: collate_pad_batch(batch, pad_idx=pad_idx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4,
                                               collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=4,
                                              collate_fn=collate)

    return train_loader, val_loader, test_loader