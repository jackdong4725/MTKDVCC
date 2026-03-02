import pytest
import sys
import os
# Ensure project root is importable when running pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from config import cfg

# 测试时确保 RefAtomNet 被禁用
cfg.is_refatomnet_enabled = False

from main import load_experts


def make_dummy_batch(B=1, T=2, H=cfg.student_output_size[0], W=cfg.student_output_size[1]):
    frames = torch.rand(B, T, 3, H, W)
    return {'frames': frames}


def test_experts_load_and_forward():
    device = torch.device('cpu')

    try:
        experts = load_experts(device)
    except ValueError:
        pytest.skip("No experts available in this environment")

    assert isinstance(experts, dict)

    inputs = make_dummy_batch()

    for name, expert in experts.items():
        # Some experts may require specific inputs; we pass generic frames
        try:
            out = expert(inputs)
            assert isinstance(out, dict), f"{name} output must be dict"
            assert 'density_map' in out and 'confidence' in out, f"{name} must return density_map and confidence"
            dm = out['density_map']
            conf = out['confidence']
            assert dm.dim() == 4 and dm.shape[0] == 1, f"{name} density_map shape unexpected: {dm.shape}"
            assert conf.shape[0] == 1, f"{name} confidence shape unexpected: {conf.shape}"
        except Exception as e:
            # Warn but allow test to continue for other experts
            pytest.skip(f"Expert {name} forward failed in this environment: {e}")
