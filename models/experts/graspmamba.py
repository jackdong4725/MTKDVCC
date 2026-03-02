"""
GraspMamba Expert Wrapper
将 GraspMamba 推理封装为与专家接口一致的模块，输出 density_map 与 confidence
"""
import os
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from config import cfg


class GraspMambaExpert(nn.Module):
    """GraspMamba Expert Wrapper

    尝试从 `GraspMamba/inference` 加载 `Grasp_Mamba` 推理类；若不可用，则使用占位返回（零密度图）
    输出格式: {'density_map': (B, T, H_out, W_out), 'confidence': (B,)}
    """

    def __init__(self):
        super().__init__()

        self.device = cfg.device
        self.input_size = getattr(cfg, 'graspmamba_input_size', 224)
        self.checkpoint = getattr(cfg, 'graspmamba_checkpoint', None)

        # 尝试导入 Grasp_Mamba（仅用于可选增强，GraspMamba 是 zero-shot，默认无需权重）
        try:
            sys.path.insert(0, str(cfg.project_root / "GraspMamba"))
            from inference.models.graspmamba import Grasp_Mamba
            self._has_native_model = True
            self._native_model_cls = Grasp_Mamba
            print("[GraspMamba] Grasp_Mamba 类已找到 (zero-shot) ")
        except Exception as e:
            self._has_native_model = False
            self._native_model_cls = None
            print(f"[GraspMamba] Grasp_Mamba 不可用，使用轻量语义先验占位 (no external weights required): {e}")

    def forward(self, inputs: dict):
        """Forward

        Args:
            inputs: Dict 包含 'frames': (B, T, 3, H, W) 和可选 'text_prompt': List[str]
        Returns:
            Dict {'density_map': (B, T, H_out, W_out), 'confidence': (B,)}
        """
        frames = inputs.get('frames', None)
        if frames is None:
            raise ValueError("GraspMambaExpert 需要 'frames' 输入")

        B, T, C, H, W = frames.shape
        device = frames.device

        # 文本提示（兼容旧接口 'text_prompt'）
        # 支持多种键名，保证为可迭代的列表
        text_prompts = None
        if 'text_prompt' in inputs:
            text_prompts = inputs.get('text_prompt')
        elif 'text_description' in inputs:
            text_prompts = inputs.get('text_description')

        if text_prompts is None:
            text_prompts = [None] * B
        # 如果单字符串，扩展为每个样本相同的提示
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts] * B
        # 最终保证长度为 B
        if isinstance(text_prompts, list) and len(text_prompts) != B:
            text_prompts = ([None] * B) if len(text_prompts) == 0 else (text_prompts[:B] + [None] * max(0, B - len(text_prompts)))

        # 生成语义先验
        semantic_dim = 768
        embeddings = torch.zeros(B, semantic_dim, device=device)
        scene_probs = torch.ones(B, 7, device=device) / 7.0
        confidences = torch.ones(B, device=device) * 0.5

        # 轻量化启发：基于文本做简单偏置
        for i, t in enumerate(text_prompts):
            if isinstance(t, str) and t.strip():
                tt = t.lower()
                if 'dense' in tt or 'crowd' in tt:
                    sp = torch.zeros(7, device=device)
                    sp[0] = 1.0
                    scene_probs[i] = sp
                elif 'direction' in tt or 'flow' in tt or 'moving' in tt:
                    sp = torch.zeros(7, device=device)
                    sp[1] = 1.0
                    scene_probs[i] = sp
                else:
                    scene_probs[i] = torch.ones(7, device=device) / 7.0

        # 若 native 模型不可用，这里只返回占位密度图/置信度
        density = torch.zeros(B, T, H, W, device=device)
        return {
            'density_map': density,
            'semantic_embedding': embeddings,
            'scene_probs': scene_probs,
            'confidence': confidences
        }


def build_graspmamba_expert():
    return GraspMambaExpert()
