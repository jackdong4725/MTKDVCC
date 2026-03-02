import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Repository root: {ROOT}")

files = [
    ('mpm_best.pth', os.path.join(ROOT, 'weights', 'mpm_best.pth')),
    ('multi_cvae_2000.pth', os.path.join(ROOT, 'weights', 'multi_cvae_2000.pth')),
    ('convnext_small_384_in22ft1k.pth', os.path.join(ROOT, 'weights', 'convnext_small_384_in22ft1k.pth')),
    ('groundingdino_swint_ogc.pth', os.path.join(ROOT, 'weights', 'groundingdino_swint_ogc.pth')),
    ('sam2.1_hiera_large.pt', os.path.join(ROOT, 'weights', 'sam2.1_hiera_large.pt')),
]

for name, path in files:
    print('----')
    print(name, '->', os.path.relpath(path, ROOT))
    exists = os.path.exists(path)
    print('exists:', exists)
    if not exists:
        continue
    try:
        print('Attempting torch.load (map_location=cpu) ...')
        obj = torch.load(path, map_location='cpu')
        print('Loaded type:', type(obj))
        if isinstance(obj, dict):
            keys = list(obj.keys())
            print('dict keys sample:', keys[:10])
            if 'model' in obj:
                if isinstance(obj['model'], dict):
                    print("'model' is dict, sample keys:", list(obj['model'].keys())[:5])
        elif torch.is_tensor(obj):
            print('Tensor shape:', obj.shape)
    except Exception as e:
        print('load failed:', repr(e))

# Also check GroundingDINO config file
gd_cfg = os.path.join(ROOT, 'CountVid', 'GroundingDINO', 'groundingdino', 'config', 'GroundingDINO_SwinB.cfg.py')
print('\nGroundingDINO config:', os.path.relpath(gd_cfg, ROOT), 'exists:', os.path.exists(gd_cfg))
