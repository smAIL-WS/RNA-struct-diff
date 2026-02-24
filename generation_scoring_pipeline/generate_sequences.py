import argparse
import os
import torch
import pickle
import sys
from pathlib import Path
# Add project root to sys.path

REPO_ROOT = Path(__file__).resolve().parent.parent   # RNA-struct-diff/
sys.path.insert(0, str(REPO_ROOT / "RNA_struct_diff"))
#from shape_guided_RNA.layers.layers import SegmentationUnet_pairwise

#layers_path = os.path.join(current_dir, "layers")



# Add it to sys.path
#if layers_path not in sys.path:
#    sys.path.insert(0, layers_path)


from shape_guided_RNA.model import get_model_class, get_model_bind, get_model_bind_RNAfold #get_model_bind_inpaint
from diffusion_utils.diffusion_multinomial import pad_length_to_valid


# Simple alphabet map for decoding
alphabet = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: "-"}

def decode_sequence(tensor):
    return ''.join(alphabet[int(tok)] for tok in tensor)

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

def strip_module_prefix(state_dict):
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}

def get_model_key_from_name(name, known_keys):
    """
    Extracts the model key (like 'bound_proj') from a full file name (e.g., 'bound_proj_1').

    Parameters:
        name (str): The full input name (e.g., a Snakefile_bckup wildcard like 'bound_proj_1').
        known_keys (List[str]): Allowed keys (e.g., ['bound_proj', 'unbound_proj']).

    Returns:
        str: The matched model key.
    """
    for key in sorted(known_keys, key=len, reverse=True):
        if name.startswith(f"{key}_") or name == key:
            return key
    raise ValueError(f"Could not match any known key in '{name}'.")

def load_model(model_key):
    checkpoint_root = REPO_ROOT / "RNA_struct_diff" / "shape_guided_RNA" / "checkpoints"

    if model_key == "2D":
        model_dir = checkpoint_root / "2D"
        model_class = get_model_bind

    elif model_key == "2D_RNAfold":
        model_dir = checkpoint_root / "2D_RNAfold"
        model_class = get_model_bind_RNAfold

    elif model_key == "1D":
        model_dir = checkpoint_root / "1D"
        model_class = get_model_bind

    else:
        raise ValueError(f"Unknown model key: {model_key}")


    with open(os.path.join(model_dir, 'args.pickle'), 'rb') as f:
        train_args = pickle.load(f)

    model = model_class(train_args)
    checkpoint = torch.load(
        os.path.join(model_dir, 'check', 'checkpoint.pt'),
        map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model.load_state_dict(strip_module_prefix(checkpoint['model']))
    model.eval().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return model

def sample_sequences(model, structure, contact_pairs, shape_factor, device, n_samples=1000):
    padded_len = pad_length_to_valid(len(structure), factor=shape_factor)
    shape = (padded_len,)
    contact_map = contact_list_to_map(contact_pairs, padded_len).to(device)
    contact_batch = contact_map.unsqueeze(0).repeat(n_samples, 1, 1)

    with torch.no_grad():
        #samples = model.sample(n_samples, shape=shape, guidance=contact_batch)
        samples = adaptive_sampling(model, contact_map, total_samples=n_samples, shape=shape,initial_batch_size=min(1000, n_samples), device=device)

    decoded = [decode_sequence(samples[i, :len(structure)]) for i in range(n_samples)]
    return decoded

def write_fasta(sequences, output_path):
    with open(output_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">sample{i}\n{seq}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, help="e.g. unbound_proj_1")
    parser.add_argument('--motif-path', type=str, required=True, help="Path to file with target dot-bracket structure")
    parser.add_argument('--output', type=str, required=True, help="Output FASTA file path")
    parser.add_argument('--n-samples', type=int, default=1000)
    args = parser.parse_args()
    known_keys = ["2D","1D","2D_RNAfold"]

    model_key = get_model_key_from_name(args.model_name,known_keys)
    model = load_model(model_key)

    with open(args.motif_path, 'r') as f:
        structure = f.read().strip()
    contact_pairs = contact_map_from_structure(structure)
    device = next(model.parameters()).device

    sequences = sample_sequences(
        model=model,
        structure=structure,
        contact_pairs=contact_pairs,
        shape_factor=4,
        device=device,
        n_samples=args.n_samples
    )

    write_fasta(sequences, args.output)
    print(f"Wrote {len(sequences)} samples to {args.output}")



def adaptive_sampling(model, contact_map, total_samples,shape, device, initial_batch_size=1000):
    """
    Samples sequences with guidance using large batches. Falls back to smaller ones on CUDA OOM.
    Assumes model.sample() handles shape and padding internally.
    """
    import torch

    samples_all = []
    remaining = total_samples
    batch_size = initial_batch_size

    while remaining > 0:
        try:
            current_batch = min(remaining, batch_size)
            contact_batch = contact_map.unsqueeze(0).repeat(current_batch, 1, 1)

            with torch.no_grad():
                chunk = model.sample(current_batch,shape=shape, guidance=contact_batch)

            samples_all.append(chunk)
            remaining -= current_batch

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise RuntimeError("Out of memory even at batch size 1 — reduce sequence length or free GPU.")
            batch_size = max(1, batch_size // 2)

    return torch.cat(samples_all, dim=0)

if __name__ == '__main__':
    main()