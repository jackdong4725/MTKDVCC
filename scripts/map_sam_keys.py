"""Heuristic key-mapping for SAM checkpoints.

Loads a checkpoint (possibly converted), applies a list of common
prefix mappings and heuristics, and writes a new checkpoint suitable
for the local SAM loader.

Usage:
    python3 scripts/map_sam_keys.py --src weights/sam_converted.pth --dst weights/sam_mapped.pth
"""
import argparse
import torch
from collections import OrderedDict

MAPPING_RULES = [
    # common trunk -> top-level
    ("image_encoder.trunk.", "image_encoder."),
    ("image_encoder.trunk.patch_embed.", "image_encoder.patch_embed."),
    ("image_encoder.trunk.pos_embed", "image_encoder.pos_embed"),
    # sam specific prefixes
    ("sam_mask_decoder.", "mask_decoder."),
    ("sam_prompt_encoder.", "prompt_encoder."),
    ("sam_", ""),
    # some checkpoints use different naming for mask_decoder/prompt_encoder
    ("mask_decoder.", "mask_decoder."),
    ("prompt_encoder.", "prompt_encoder."),
    # image encoder variations
    ("image_encoder.trunk.", "image_encoder."),
    ("image_encoder_neck.", "image_encoder.neck."),
]

# Extra heuristics to try if direct mappings fail
def heuristic_transform(k):
    # remove leading namespaces like "model." or "module."
    if k.startswith('model.'):
        k = k[len('model.'):]
    if k.startswith('module.'):
        k = k[len('module.'):]
    # replace double dots
    k = k.replace('..', '.')
    return k


def map_keys(state_dict):
    mapped = OrderedDict()
    mapped_examples = []

    for k, v in state_dict.items():
        new_k = k
        # apply mapping rules in order
        for src_pref, dst_pref in MAPPING_RULES:
            if new_k.startswith(src_pref):
                new_k = dst_pref + new_k[len(src_pref):]
                break
        # additional heuristic
        new_k = heuristic_transform(new_k)
        mapped[new_k] = v
        if new_k != k and len(mapped_examples) < 20:
            mapped_examples.append((k, new_k))

    return mapped, mapped_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=False)
    args = parser.parse_args()

    src = args.src
    dst = args.dst or src.replace('.pth', '_mapped.pth')

    print('Loading:', src)
    ckpt = torch.load(src, map_location='cpu')
    # extract dict like previous converter
    if isinstance(ckpt, dict):
        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            state = ckpt['model']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state = ckpt['state_dict']
        else:
            # assume top-level is state dict
            state = ckpt
    else:
        raise RuntimeError('Unsupported checkpoint format')

    # strip module. prefix
    state = { (k[len('module.'): ] if k.startswith('module.') else k):v for k,v in state.items() }

    mapped, examples = map_keys(state)

    print('Mapped keys:', len(mapped), 'original keys:', len(state))
    print('\nExamples of remapped keys (original -> mapped):')
    for a,b in examples:
        print('  ', a, '->', b)

    # Save mapped checkpoint
    torch.save(mapped, dst)
    print('Saved mapped checkpoint to', dst)


if __name__ == '__main__':
    main()
