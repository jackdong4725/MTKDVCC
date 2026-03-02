import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg

try:
    import torch
except Exception:
    pytest.skip("torch not available in this environment", allow_module_level=True)


WEIGHTS = [
    ('mpm_best.pth', os.path.join('weights', 'mpm_best.pth')),
    ('multi_cvae_2000.pth', os.path.join('weights', 'multi_cvae_2000.pth')),
    ('convnext_small_384_in22ft1k.pth', os.path.join('weights', 'convnext_small_384_in22ft1k.pth')),
    ('groundingdino_swint_ogc.pth', os.path.join('weights', 'groundingdino_swint_ogc.pth')),
    ('sam2.1_hiera_large.pt', os.path.join('weights', 'sam2.1_hiera_large.pt')),
]


def test_weights_exist():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    missing = []
    for name, rel in WEIGHTS:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            missing.append(rel)
    if missing:
        pytest.skip(f"Missing weight files: {missing}")


def test_weights_torch_loadable():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results = {}

    for name, rel in WEIGHTS:
        p = os.path.join(root, rel)
        try:
            obj = torch.load(p, map_location='cpu')
            results[rel] = ('ok', type(obj))
        except Exception as e:
            # PyTorch 2.6 may raise a Weights-only / safe globals error for some checkpoints (GroundingDINO)
            # Try a secondary load with weights_only=False to check if it can be fully loaded (trusted local file)
            try:
                obj = torch.load(p, map_location='cpu', weights_only=False)
                results[rel] = ('ok_with_weights_only_false', type(obj))
            except Exception as e2:
                results[rel] = ('error', repr(e2))

    # Print summary for user
    for k, v in results.items():
        print(f"{k}: {v[0]} - {v[1]}")

    # Treat both primary success and success with weights_only=False as acceptable
    acceptable = ['ok', 'ok_with_weights_only_false']
    errs = [k for k, v in results.items() if v[0] not in acceptable]
    if errs:
        pytest.fail(f"Some weight files failed to torch.load: {errs}")

    # Inform about any files that required weights_only=False
    special = [k for k, v in results.items() if v[0] == 'ok_with_weights_only_false']
    if special:
        print('\nNote: The following checkpoints loaded only when using torch.load(..., weights_only=False):', special)
        print('This indicates the checkpoint contains objects that require the legacy unpickling behavior (e.g., argparse.Namespace).')
        print('To load them in modern PyTorch safely, either use trusted environment or load with weights_only=False if you trust the source.')
