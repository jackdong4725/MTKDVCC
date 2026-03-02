import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from models.experts.graspmamba import GraspMambaExpert


def test_graspmamba_forward_shapes():
    model = GraspMambaExpert()
    model.eval()

    B, T, C, H, W = 2, 4, 3, 192, 192
    frames = torch.rand(B, T, C, H, W)

    out = model({'frames': frames, 'text_prompt': ['crowd', 'sparse']})

    assert 'semantic_embedding' in out
    assert 'scene_probs' in out
    assert out['semantic_embedding'].shape == (B, 768)
    assert out['scene_probs'].shape == (B, 7)
    assert out['confidence'].shape == (B,)
