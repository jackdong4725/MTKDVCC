"""Utility to ensure the repository has a SAM checkpoint compatible with
its current model code.

The CountVid expert (and possibly other code) expects a specific SAM variant
(`vit_h` by default) and will call ``sam_model_registry[variant]`` when
initializing.  If the checkpoint at ``cfg.sam_checkpoint`` does not match,
loading will raise size-mismatch errors (as we observed with a hiera_large
file whose hidden dimension was 144 instead of 1280).

This script does the following:

1. Reads ``cfg.sam_model_type`` and ``cfg.sam_checkpoint`` from the project
   config.
2. Attempts to instantiate that SAM model using the registry and the
   existing checkpoint.
3. If loading fails (or the checkpoint is missing), downloads a known-good
   checkpoint URL for the requested variant and retries.
4. Optionally converts the checkpoint to the "flat" format expected by the
   repository (strip wrappers, remove `module.` prefixes).  This is the same
   conversion we perform in ``scripts/convert_sam_ckpt.py`` when necessary.

Usage::

    python scripts/fetch_compatible_sam.py [-v]

You can also call this from your training/evaluation entrypoint before
calling :func:`load_experts` if you want the workflow to self-heal.
"""

import argparse
import sys
import os
from pathlib import Path

import torch

# make sure project root is on path so config import works
proj_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

from config import cfg

# known good urls for official SAM release weights
KNOWN_SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_t": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_t_01ec64.pth",
}


def download_checkpoint(variant: str, dest: Path) -> None:
    """Download the checkpoint for *variant* to *dest*.

    Raises ``KeyError`` if we don't know a URL for the variant.  The call
    clobbers *dest* if it already exists.
    """
    if variant not in KNOWN_SAM_URLS:
        raise KeyError(f"no known URL for SAM variant '{variant}'")
    url = KNOWN_SAM_URLS[variant]
    print(f"downloading {variant} checkpoint from {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    urllib.request.urlretrieve(url, dest)
    print("download complete")


def try_load(variant: str, ckpt_path: Path):
    """Attempt to build a SAM model of the given *variant* using
    *ckpt_path*.  Raises whatever exception the registry throws or returns the
    model if successful.  ``model.eval()`` is called internally so it is ready
    to use.
    """
    try:
        from segment_anything import sam_model_registry
    except ImportError as e:
        raise RuntimeError("segment_anything package not available") from e

    if variant not in sam_model_registry:
        raise RuntimeError(f"variant '{variant}' not found in sam_model_registry")
    model = sam_model_registry[variant](checkpoint=str(ckpt_path))
    model.eval()
    return model


def convert_to_flat(src: Path, dst: Path) -> None:
    """Copy/convert a checkpoint that may be wrapped in a ``{'model': ...}``
    dict into a plain state dict.  Also strips ``'module.'`` prefixes.  This
    mirrors :mod:`scripts.convert_sam_ckpt` but we include it here so the
    workflow is self contained.
    """
    ckpt = torch.load(src, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    # strip module.
    new = {}
    for k, v in ckpt.items():
        nk = k.replace("module.", "")
        new[nk] = v
    torch.save(new, dst)
    print(f"converted {src} -> {dst} (flat state dict)")


def main(verbose: bool = False):
    variant = cfg.sam_model_type
    ckpt_path = Path(cfg.sam_checkpoint)
    print(f"config expects SAM variant {variant}")

    if not ckpt_path.exists():
        print(f"checkpoint {ckpt_path} does not exist")
        download_checkpoint(variant, ckpt_path)

    # try loading; on failure we'll attempt a download and retry
    try:
        _ = try_load(variant, ckpt_path)
        print("existing checkpoint loaded successfully")
        return
    except Exception as e:
        print(f"failed to load existing checkpoint: {e}")
        if verbose:
            import traceback

            traceback.print_exc()

    # attempt download and retry
    try:
        download_checkpoint(variant, ckpt_path)
        _ = try_load(variant, ckpt_path)
        print("downloaded & loaded checkpoint successfully")
        return
    except Exception as e:
        print(f"downloaded checkpoint still failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        print("please inspect cfg.sam_model_type or provide a compatible file manually")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensure a compatible SAM checkpoint")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    main(verbose=args.verbose)
