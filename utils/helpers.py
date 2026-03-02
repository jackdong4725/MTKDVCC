"""
辅助工具函数
"""
import torch
import random
import numpy as np


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model, input_shape, device='cuda'):
    """
    计算模型FLOPs（仅支持单输入模型）

    Args:
        model: nn.Module
        input_shape: tuple, e.g., (1, 8, 3, 768, 768)
        device: str
    Returns:
        flops, params
    """
    try:
        from thop import profile
        model = model.to(device)
        dummy_input = torch.randn(input_shape).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except ImportError:
        print("[警告] 请安装 thop: pip install thop")
        return None, None
    except Exception as e:
        print(f"[警告] FLOPs 计算失败: {e}")
        return None, None