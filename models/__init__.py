"""
Models package - 支持预训练和随机初始化
"""

import torch
import torch.nn as nn
import os
import collections # ✅ 新增

from config import cfg


def load_model(model_name, device='cuda'):
    """
    动态加载模型
    """
    model_name_lower = model_name.lower()

    # ==================== 创建模型 ====================
    model = None # 初始化 model
    if model_name_lower == 'pointdgmambastudent': # ✅ 学生模型
        from .student.pointdgmamba import PointDGMambaStudent
        model = PointDGMambaStudent()
        print(f"  ✅ Loaded PointDGMambaStudent: {model.count_parameters() / 1e6:.2f}M params")

    elif model_name_lower == 'scene_classifier':
        from .scene_classifier import SceneClassifier
        model = SceneClassifier()
        print(f"  ✅ Loaded Scene Classifier")

    # === 专家模型 ===
    elif model_name_lower == 'countvid': # ✅ CountVid
        from .experts.countvid import CountVid
        model = CountVid()
        print(f"  ✅ Created CountVid")

    elif model_name_lower == 'graspmamba': # ✅ GraspMamba (zero-shot)
        from .experts.graspmamba import GraspMambaExpert as GraspMamba
        model = GraspMamba()
        print(f"  ✅ Created GraspMamba (zero-shot)")


    elif model_name_lower == 'crowdmpm': # ✅ CrowdMPM
        from .experts.crowdmpm import CrowdMPM
        model = CrowdMPM()
        print(f"  ✅ Created CrowdMPM")

    elif model_name_lower == 'oman': # ✅ OMAN
        from .experts.oman import OMAN
        model = OMAN()
        print(f"  ✅ Created OMAN")

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ==================== 加载预训练权重（如果有）====================
    # ✅ 根据 model_name 获取对应的权重路径
    weight_path = None
    if model_name_lower == 'countvid':
        # CountVid 依赖 GroundingDINO 和 SAM，在 CountVid 内部加载，这里不处理
        pass
    elif model_name_lower == 'graspmamba':
        # GraspMamba is zero-shot by default; no external checkpoint is required
        weight_path = None
    elif model_name_lower == 'crowdmpm':
        # CrowdMPM 权重在 CrowdMPM 类内部统一处理，这里不处理
        pass
    elif model_name_lower == 'oman':
        weight_path = cfg.oman_checkpoint
    elif model_name_lower == 'pointdgmambastudent': # 学生模型不加载预训练权重，而是训练
        pass
    elif model_name_lower == 'scene_classifier': # 场景分类器通常从头训练或依赖其他预训练模型
        pass

    if weight_path and os.path.exists(weight_path):
        try:
            print(f"  尝试加载 {model_name} 权重文件: {weight_path}")
            checkpoint = torch.load(weight_path, map_location='cpu')

            state_dict = None
            if isinstance(checkpoint, dict):
                # 尝试从常用键中提取 state_dict
                for key in ['model', 'state_dict', 'net', 'network']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                # 如果没有找到，假设checkpoint本身就是state_dict
                if state_dict is None:
                    state_dict = checkpoint
            else:
                # 如果checkpoint不是字典，假设它就是state_dict
                state_dict = checkpoint

            # 移除 DataParallel 或 DistributedDataParallel 添加的 'module.' 前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print(f"  ✅ Loaded pretrained weights from {weight_path}")
            else:
                print(f"  ⚠️  Loaded pretrained weights from {weight_path} with mismatches:")
                if missing_keys: print(f"      Missing Keys: {missing_keys[:5]}...")
                if unexpected_keys: print(f"      Unexpected Keys: {unexpected_keys[:5]}...")

        except Exception as e:
            print(f"  ⚠️  Failed to load weights from {weight_path}: {e}")
            print(f"  ℹ️  Using random initialization for {model_name}")

    else:
        # ✅ 学生模型和场景分类器可能不需要预训练权重或在内部加载
        if model_name_lower not in ['pointdgmambastudent', 'scene_classifier', 'countvid', 'crowdmpm']:
            print(f"  ⚠️  No specific pretrained weights found for {model_name} at {weight_path}")
            print(f"  ℹ️  Using random initialization for {model_name}")


    # ==================== 移动到设备 ====================
    if model is not None:
        model = model.to(device)

        # ==================== 冻结教师模型 ====================
        # 学生模型和场景分类器不冻结，专家模型冻结
        if model_name_lower not in ['pointdgmambastudent', 'scene_classifier']:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            print(f"  🔒 {model_name} parameters frozen (teacher mode)")

        return model
    else:
        raise ValueError(f"Model {model_name} could not be initialized.")


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
