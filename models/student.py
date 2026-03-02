"""
Models package - 支持预训练和随机初始化
"""

import torch
import torch.nn as nn
import os
from config import Config


def load_model(model_name, device='cuda'):
    """
    动态加载模型
    """
    model_name_lower = model_name.lower()

    # ==================== 创建模型 ====================

    if model_name_lower == 'vmamba':
        from .vmamba import VMambaCounter
        model = VMambaCounter()
        print(f"  ✅ Loaded VMamba: {model.count_parameters() / 1e6:.2f}M params")

    elif model_name_lower == 'scene_classifier':
        from .scene_classifier import SceneClassifier
        model = SceneClassifier(num_classes=len(Config.datasets))
        print(f"  ✅ Loaded Scene Classifier")

    elif model_name_lower == 'sasnet':
        from .sasnet import SASNet
        model = SASNet()
        print(f"  ✅ Created SASNet")

    elif model_name_lower == 'p2pnet':
        from .p2pnet import P2PNet
        model = P2PNet()
        print(f"  ✅ Created P2PNet")

    elif model_name_lower == 'clip_ebc':
        from .clip_ebc import CLIPEBC
        model = CLIPEBC()
        print(f"  ✅ Created CLIP-EBC")

    elif model_name_lower == 'crowdclip':
        from .crowdclip import CrowdCLIP
        model = CrowdCLIP()
        print(f"  ✅ Created CrowdCLIP")

    elif model_name_lower == 'mobilecount':
        from .mobilecount import MobileCount
        model = MobileCount()
        print(f"  ✅ Created MobileCount")

    # 🔥 新增：DACM
    elif model_name_lower == 'dacm':
        from .dacm import DACM
        model = DACM()
        print(f"  ✅ Created DACM: {model.count_parameters() / 1e6:.2f}M params")

    # 🔥 新增：DenseVmamba
    elif model_name_lower == 'dense_vmamba':
        from .dense_vmamba import DenseVmamba
        model = DenseVmamba()
        print(f"  ✅ Created DenseVmamba: {model.count_parameters() / 1e6:.2f}M params")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ==================== 加载预训练权重（如果有）====================
    weight_path = Config.pretrained_paths.get(model_name_lower)

    if weight_path and os.path.exists(weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print(f"  ✅ Loaded pretrained weights from {weight_path}")
            else:
                print(f"  ⚠️  Loaded pretrained weights with mismatches")

        except Exception as e:
            print(f"  ⚠️  Failed to load weights: {e}")
            print(f"  ℹ️  Using random initialization")

    else:
        if model_name_lower not in ['vmamba', 'scene_classifier']:
            print(f"  ⚠️  No pretrained weights for {model_name}")
            print(f"  ℹ️  Using random initialization")

    # ==================== 移动到设备 ====================
    model = model.to(device)

    # ==================== 冻结教师模型 ====================
    if model_name_lower not in ['vmamba', 'scene_classifier']:
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        print(f"  🔒 {model_name} parameters frozen (teacher mode)")

    return model

def initialize_weights(model):
    """
    为模型应用 Kaiming 初始化（用于随机初始化的教师）
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


__all__ = ['load_model', 'initialize_weights']
