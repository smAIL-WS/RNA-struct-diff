import os
import argparse
import pickle
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from model import get_model_class
from diffusion_utils.diffusion_multinomial import pad_length_to_valid
import collections
RNA_SS_data = collections.namedtuple('RNA_SS_data', 'data_fcn_2 seq_raw length name contact')
sys.modules['__main__'].RNA_SS_data = RNA_SS_data

PAD_TOKEN = 4
#alphabet = {0: 'A', 1: 'U', 2: 'C', 3: 'G', 4: '-'}
alphabet = {0:'A', 1:'C', 2: 'G', 3: 'U', 4: "-"}


def decode_sequence(tensor):
    return [''.join(alphabet[int(tok)] for tok in sample.view(-1)) for sample in tensor]
def save_fasta(samples_per_entry, output_path, is_chain=False):
    with open(output_path, 'w') as f:
        for name, samples in samples_per_entry:
            if is_chain:
                for step, step_samples in enumerate(samples):
                    for i, seq in enumerate(decode_sequence(step_samples)):
                        f.write(f">Sample_{name}_step{step}_i{i}\n{seq}\n")
                    f.write("\n")
            else:
                for i, seq in enumerate(decode_sequence(samples)):
                    f.write(f">{name}_sample{i}\n{seq}\n")


def contact_list_to_map(contact_list, length):
    mat = torch.zeros((length, length), dtype=torch.float32)
    for i, j in contact_list:
        mat[i, j] = 1
        mat[j, i] = 1
    return mat

import re

def parse_txt_input(path):
    entries = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # Skip empty lines

    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            # Parse name and length
            header = lines[i][1:]
            match = re.match(r'(\S+)\s+len=(\d+)', header)
            if not match:
                raise ValueError(f"Invalid header format: {lines[i]}")
            name = match.group(1)
            length = int(match.group(2))

            # Ensure we have enough lines
            if i + 2 >= len(lines):
                raise ValueError(f"Incomplete entry starting at line {i}: {lines[i]}")

            sequence = lines[i + 1]  # not used, but you can keep it if needed

            # Parse contacts
            contact_line = lines[i + 2]
            if not contact_line.startswith("; contacts:"):
                raise ValueError(f"Expected contacts line at line {i+2}, got: {contact_line}")

            contact_numbers = list(map(int, contact_line.replace("; contacts:", "").split()))
            if len(contact_numbers) % 2 != 0:
                raise ValueError("Odd number of contact indices")

            contacts = [(contact_numbers[j], contact_numbers[j + 1]) for j in range(0, len(contact_numbers), 2)]

            entries.append((name, length, contacts))
            i += 3
        else:
            raise ValueError(f"Unexpected line at {i}: {lines[i]}")
    print(entries[0])
    return entries



'''
    entries = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            name = lines[i][1:]
            length = int(lines[i + 1])
            contacts = []
            i += 2
            while i < len(lines) and not lines[i].startswith('>'):
                parts = lines[i].split()
                if len(parts) == 2:
                    contacts.append((int(parts[0]), int(parts[1])))
                i += 1
            entries.append((name, length, contacts))
        else:
            raise ValueError(f"Unexpected line: {lines[i]}")
    return entries
'''
def parse_json_input(path):
    import json
    with open(path, 'r') as f:
        entries = json.load(f)
    parsed = []
    for item in entries:
        name = item.get("name", "unnamed")
        length = item["length"]
        contacts = [tuple(pair) for pair in item["contacts"]]
        parsed.append((name, length, contacts))
    return parsed

def parse_pickle_input(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    entries = []
    for entry in data:
        name = getattr(entry, 'name', 'unnamed')
        length = len(entry.seq_raw)
        contacts = entry.contact
        entries.append((name, length, contacts))
    return entries

def strip_module_prefix(state_dict):
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}


# === Memory-aware sampling utilities ===

MEMORY_BUDGET = int(720_000 * 2.5)  # from 200×60² training baseline

def estimate_sample_cost(length, n_samples):
    """Rough memory cost: O(length² × n_samples)"""
    return (length ** 2) * n_samples

def estimate_max_safe_batch(length):
    """Return the largest safe number of samples for a given sequence length"""
    return max(1, MEMORY_BUDGET // (length ** 2))

def safe_sample_entry(model, name, length, contact_list, n_samples, shape_factor, device, output_path, is_chain):
    from diffusion_utils.diffusion_multinomial import pad_length_to_valid

    padded_len = pad_length_to_valid(length, factor=shape_factor)
    shape = (padded_len,)
    max_batch = estimate_max_safe_batch(length)

    contact_map_full = contact_list_to_map(contact_list, padded_len).to(device)

    samples_all = []

    for i in range(0, n_samples, max_batch):
        this_batch = min(max_batch, n_samples - i)
        contact_batch = contact_map_full.unsqueeze(0).repeat(this_batch, 1, 1)

        with torch.no_grad():
            if is_chain:
                chunk = model.sample_chain(this_batch, shape, guidance=contact_batch)
            else:
                chunk = model.sample(this_batch, shape, guidance=contact_batch)

        samples_all.append(chunk)

    if is_chain:
        full = torch.cat(samples_all, dim=1)[:, :, :length]
    else:
        full = torch.cat(samples_all, dim=0)[:, :length]

    append_fasta(
        name=name,
        samples=full,
        output_path=output_path,
        contacts=contact_list,
        length=length,
        n_samples=n_samples,
        is_chain=is_chain
    )
def main():
    MEMORY_BUDGET = int(720_000 * 2.5)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--format', choices=['pickle', 'txt'], default='pickle')
    parser.add_argument('--n-samples', type=int, default=16)
    parser.add_argument('--chain', type=eval, default=False)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--diffusion-class', choices=['standard', 'bound_token',"bound_rnafold"], default='standard',
                    help='Choose which diffusion class to use')
    args = parser.parse_args()

    # Load model
    with open(os.path.join(args.model, 'args.pickle'), 'rb') as f:
        train_args = pickle.load(f)


    if args.diffusion_class == 'bound_token':
        from model import get_model_bind
        model = get_model_bind(train_args)
    elif args.diffusion_class == 'bound_rnafold':
        from model import get_model_bind_RNAfold
        model = get_model_bind_RNAfold(train_args)
    else:
        from model import get_model_class
        model = get_model_class(train_args)

    checkpoint = torch.load(os.path.join(args.model, 'check', 'checkpoint.pt'), map_location='cpu')
    #because of DDP, it is wrapped different (with module. .... )
    model.load_state_dict(strip_module_prefix(checkpoint['model']))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    device = next(model.parameters()).device
    pad_factor = 4

    # Load input
    if args.format == 'pickle':
        entries = parse_pickle_input(args.input)
    elif args.format == 'txt':
        entries = parse_txt_input(args.input)
    elif args.format == 'json':
        entries = parse_json_input(args.input)
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    all_samples = []
    output_path = args.output or os.path.join(args.model, 'samples')

# Treat it as a directory unless it ends with .fasta or .txt etc.

    output_path = args.output or os.path.join(args.model, 'samples', f'structured_samples.fasta')

    if not os.path.splitext(output_path)[1]:  # no file extension → treat as dir
        os.makedirs(output_path, exist_ok=True)
        output_dir = output_path
        output_path = os.path.join(output_dir, 'structured_samples.fasta')  # or any default file name
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    for name, length, contact_list in entries:
        safe_sample_entry(
            model=model,
            name=name,
            length=length,
            contact_list=contact_list,
            n_samples=args.n_samples,
            shape_factor=pad_factor,
            device=device,
            output_path=output_path,
            is_chain=args.chain
        )


'''
    for name, length, contact_list in entries:
        padded_len = pad_length_to_valid(length, factor=pad_factor)
        contact_map = contact_list_to_map(contact_list, padded_len).to(device)
        contact_map = contact_map.unsqueeze(0).repeat(args.n_samples, 1, 1)
        shape = (padded_len,)

        with torch.no_grad():
            if args.chain:
                samples = model.sample_chain(args.n_samples, shape, guidance=contact_map)
                #all_samples.append((name, samples[:, :, :length]))# keep only final step
                append_fasta(name,
                 samples[:, :length],
                 output_path,
                 contacts=contact_list,
                 length=length,
                 n_samples=args.n_samples,
                 is_chain=args.chain
                 )

            else:
                samples = model.sample(args.n_samples, shape, guidance=contact_map)
                #all_samples.append((name, samples[:, :length]))
                #append_fasta(name, samples[:, :length], output_path, is_chain=args.chain)
                append_fasta(name,
                             samples[:, :length],
                             output_path,
                             contacts=contact_list,
                             length=length,
                             n_samples=args.n_samples,
                             is_chain=args.chain
                             )

    # Save
    #output_path = args.output or os.path.join(args.model, 'samples', f'structured_samples.fasta')
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #save_fasta(all_samples, output_path, is_chain=args.chain)
    print(f"Saved samples to {output_path}")
'''
def append_fasta(name, samples, output_path, contacts, length, n_samples, is_chain=False):
    def contact_string(clist):
        return ' '.join(f"{i} {j}" for i, j in clist)

    with open(output_path, 'a') as f:
        if is_chain:
            for step, step_samples in enumerate(samples):
                for i, seq in enumerate(decode_sequence(step_samples)):
                    f.write(f">Sample_{name}_step{step}_i{i} len={length}\n")
                    f.write(f"{seq}\n")
                    f.write(f"; contacts: {contact_string(contacts)}\n")
                f.write("\n")
        else:
            for i, seq in enumerate(decode_sequence(samples)):
                f.write(f">{name}_sample{i} len={length}\n")
                f.write(f"{seq}\n")
                f.write(f"; contacts: {contact_string(contacts)}\n")

if __name__ == '__main__':
    main()
