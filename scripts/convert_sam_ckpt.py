"""Convert SAM checkpoint to a flat state_dict expected by local SAM loader.

Usage:
    python scripts/convert_sam_ckpt.py --src /path/to/sam_checkpoint.pt --dst /path/to/sam_converted.pth

If --dst omitted, saves to same folder as src with suffix _converted.pth
"""
import argparse
import torch
import os


def load_and_extract(src_path):
    ckpt = torch.load(src_path, map_location='cpu')
    # Common wrapping patterns
    if isinstance(ckpt, dict):
        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            state = ckpt['model']
            print('Found top-level "model" key; using ckpt["model"] as state_dict')
            return state
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state = ckpt['state_dict']
            print('Found "state_dict" key; using ckpt["state_dict"] as state_dict')
            return state
        # some checkpoints store under ['model_state_dict'] or ['net']
        for alt in ['model_state_dict', 'net', 'state', 'weights']:
            if alt in ckpt and isinstance(ckpt[alt], dict):
                print(f'Found "{alt}" key; using ckpt["{alt}"] as state_dict')
                return ckpt[alt]
        # If keys look already like SAM keys (contain a dot), assume it's ready
        sample_keys = list(ckpt.keys())[:10]
        if any('.' in k for k in sample_keys):
            print('Top-level keys appear to be a state_dict already; using checkpoint as-is')
            return ckpt
    # Fallback: cannot interpret
    raise RuntimeError('Unrecognized checkpoint format; no model/state_dict-like key found')


def strip_module_prefix(state_dict):
    # remove leading 'module.' if present
    new = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith('module.'):
            new_k = k[len('module.'):]
        new[new_k] = v
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=False)
    args = parser.parse_args()

    src = args.src
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    dst = args.dst
    if dst is None:
        base, ext = os.path.splitext(src)
        dst = base + '_converted.pth'

    print(f'Loading {src}...')
    state = load_and_extract(src)
    # Strip common wrappers
    state = strip_module_prefix(state)

    # Quick heuristic check: require some SAM-like keys
    required_substrs = ['image_encoder', 'prompt_encoder', 'mask_decoder']
    if not any(s in k for s in required_substrs for k in state.keys()):
        print('Warning: converted state_dict may not contain typical SAM keys. Still saving.')

    print(f'Saving converted checkpoint to {dst} ...')
    torch.save(state, dst)
    print('Done')


if __name__ == '__main__':
    main()
