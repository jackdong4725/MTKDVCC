"""
难度感知模块
修复：接口对齐、Top-2分歧计算、学生不确定度
"""
import torch
import torch.nn.functional as F
from config import cfg


class DifficultyAwareModule:
    """难度感知模块"""

    def __init__(self):
        self.difficulty_ema = None

    def compute_difficulty(self, frames, teacher_outputs_list, student_output, device):
        """
        计算样本难度（教师分歧 + 学生不确定度）

        Args:
            frames: (B, T, 3, H, W)
            teacher_outputs_list: List[Dict] - 每样本的教师输出字典
            student_output: Dict - 学生输出
            device: torch.device
        Returns:
            difficulty_scores: (B,)
        """
        B = frames.shape[0]

        # 1. 教师分歧度（仅对有多个教师的样本计算）
        divergence = torch.zeros(B, device=device)

        for b in range(B):
            sample_teachers = teacher_outputs_list[b]
            if len(sample_teachers) < 2:
                continue

            densities = [out['density_map'] for out in sample_teachers.values()]
            stacked = torch.stack(densities, dim=0)  # (K, 1, T, H, W)

            # 方差作为分歧度
            var = stacked.var(dim=0).mean()
            divergence[b] = var

        # 2. 学生不确定度（密度图方差）
        student_density = student_output['density_map']
        student_uncertainty = student_density.var(dim=[1, 2, 3])  # (B,)

        # 3. 综合难度
        difficulty = divergence + student_uncertainty

        # 归一化到 [0, 1]
        if difficulty.max() > difficulty.min():
            difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min() + 1e-8)

        # EMA平滑
        if self.difficulty_ema is None:
            self.difficulty_ema = difficulty
        else:
            self.difficulty_ema = cfg.difficulty_ema_alpha * self.difficulty_ema + \
                                  (1 - cfg.difficulty_ema_alpha) * difficulty

        return self.difficulty_ema.clamp(0, 1)