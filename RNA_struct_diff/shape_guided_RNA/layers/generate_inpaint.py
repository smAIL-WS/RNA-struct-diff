import torch
import yaml
import os
import sys
from model import get_model_bind_inpaint, get_model_bind_inpaint_1D
from diffusion_utils.diffusion_multinomial import MultinomialDiffusion
import pickle
from diffusion_utils.diffusion_multinomial import pad_length_to_valid
alphabet = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'N': -1}
rev_alphabet = {v: k for k, v in alphabet.items()}
rev_alphabet[4] = "-"  # fallback if padding token is used


def decode_sequence(tensor):
    alphabet = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: "-"}
    return ''.join(alphabet[int(tok)] for tok in tensor)


def strip_module_prefix(state_dict):
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}
def contact_list_to_map(contact_list, length):
    mat = torch.zeros((length, length), dtype=torch.float32)
    for i, j in contact_list:
        mat[i, j] = 1
        mat[j, i] = 1
    return mat
def encode_input(seq_str):
    """Convert sequence like 'ACGN' into LongTensor of indices"""
    return torch.tensor([alphabet.get(ch, -1) for ch in seq_str], dtype=torch.long)

def make_inpaint_mask(encoded_seq):
    """1 where fixed (not N), -1 where to inpaint"""
    return (encoded_seq != -1).long()

def pad_tensor(tensor, max_len, pad_value=4):
    """Pad tensor to max_len"""
    padded = torch.full((max_len,), pad_value, dtype=tensor.dtype)
    padded[:len(tensor)] = tensor
    return padded



def load_inpaint_model(model_dir, device='cuda'):
    # Load training arguments
    args_path = os.path.join(model_dir, 'args.pickle')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)

    # Build model
    #model = get_model_bind_inpaint(args)

    model = get_model_bind_inpaint_1D(args)


    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, 'check', 'checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(strip_module_prefix(checkpoint['model']))

    # Finalize model
    model.eval()
    model.to(device)
    return model, args


def contact_map_from_structure(structure):
    stack = []
    contacts = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            contacts.append((j, i))
    return contacts

def contact_list_to_map(contact_list, length):
    mat = torch.zeros((length, length), dtype=torch.float32)
    for i, j in contact_list:
        mat[i, j] = 1
        mat[j, i] = 1
    return mat

def sample_multiple_from_single(model, args, seq, struct, n_samples=8):
    device = next(model.parameters()).device
    seq_tensor, mask_tensor = encode_sequence(seq, pad_token=args.pad_token)
    L = len(seq_tensor)
    padded_len = pad_length_to_valid(L, factor=SHAPE_FACTOR)
    pad_size = padded_len - L

    # Pad sequence and mask
    seq_tensor = torch.cat([seq_tensor, torch.full((pad_size,), args.pad_token)])
    mask_tensor = torch.cat([mask_tensor, torch.zeros(pad_size, dtype=torch.long)])

    # Encode contact map from structure
    contacts = contact_map_from_structure(struct)
    contact_map = contact_list_to_map(contacts, padded_len).to(device)
    contact_batch = contact_map.unsqueeze(0).repeat(n_samples, 1, 1)

    # Prepare batched inputs
    seq_batch = seq_tensor.unsqueeze(0).repeat(n_samples, 1).to(device)
    mask_batch = mask_tensor.unsqueeze(0).repeat(n_samples, 1).to(device)
    with torch.no_grad():
        samples = model.sample(
            n_samples,
            shape=(padded_len,),
            guidance=contact_batch,
            sequence=seq_batch,
            inpaint_mask=mask_batch
        )

    return [decode_sequence(samples[i][:L]) for i in range(n_samples)]

def encode_sequence(seq, vocab="ACGUT", pad_char='N', pad_token=4):
    vocab_map = {ch: i for i, ch in enumerate(vocab)}
    indices = []
    mask = []
    for ch in seq:
        if ch in vocab_map:
            indices.append(vocab_map[ch])
            mask.append(1)
        else:
            indices.append(pad_token)
            mask.append(0)
    return torch.tensor(indices), torch.tensor(mask)

model_path = "/home/gback/discrete_diffusion/shape_guided_RNA/log/flow/all/multinomial_diffusion/multistep/inpaint_1D_unaware_es"
model, args = load_inpaint_model(model_path,device="cuda:0")

#model = load_diffusion_model(config_path, checkpoint_path)

seq = "NCGUUCGCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"
struct = ".....((((((.((((((((....))))))).)))((((((((((....))))))))))))))....."
N_SAMPLES = 1000
SHAPE_FACTOR = 4
samples = sample_multiple_from_single(model, args, seq, struct, n_samples=1000)
with open("/home/gback/discrete_diffusion/compare_structure/intermediate/hard_Y/inpaint.fasta","w") as f:

    for i, s in enumerate(samples):
        f.writelines(">sample"+str(i)+"\n"+s+"\n")
        print(f"Sample {i+1}: {s}")