"""
元教师生成器 (Meta-Teacher Generator)
理论支柱 1：双层学习 - 作为外层优化器，学习"如何教学"

核心思想：
- 输入：场景特征 + 学生当前状态
- 输出：虚拟教师参数 ξ，用于动态调整教学策略
"""
import torch
import torch.nn as nn
from config import cfg


class MetaTeacherGenerator(nn.Module):
    """
    元教师生成器：生成虚拟教师参数，指导学生学习

    输入：
        - scene_features: (B, 135) - 场景分类器的输入特征
        - student_state: (B, D_student) - 学生模型的中间状态

    输出：
        - xi: (B, meta_output_dim) - 虚拟教师参数
    """

    def __init__(self, student_feature_dim=768):
        super().__init__()

        self.input_dim = cfg.meta_input_dim + student_feature_dim

        # 元教师网络
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in cfg.meta_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, cfg.meta_output_dim))
        self.network = nn.Sequential(*layers)

        # 用于动态调整 KL 散度权重的子网络
        self.lambda_generator = nn.Sequential(
            nn.Linear(cfg.meta_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, scene_features, student_state):
        """
        Args:
            scene_features: (B, 135)
            student_state: (B, D_student)

        Returns:
            xi: (B, meta_output_dim) - 虚拟教师参数
            lambda_kl: (B, 1) - 动态 KL 散度权重
        """
        # 拼接输入
        combined = torch.cat([scene_features, student_state], dim=1)  # (B, input_dim)

        # 生成虚拟教师参数
        xi = self.network(combined)  # (B, meta_output_dim)

        # 生成动态权重
        lambda_kl = self.lambda_generator(xi) * cfg.lambda_kl_initial  # (B, 1)

        return xi, lambda_kl

    def get_teaching_signal(self, xi, teacher_outputs):
        """
        根据 ξ 参数，动态调整多个教师的输出

        Args:
            xi: (B, meta_output_dim)
            teacher_outputs: Dict[str, Tensor] - 各教师的输出

        Returns:
            adjusted_outputs: Dict[str, Tensor] - 调整后的教师输出
        """
        # 示例：用 ξ 的前K维作为各教师的权重
        num_teachers = len(teacher_outputs)
        if num_teachers > 0:
            weights = xi[:, :num_teachers]  # (B, K)
            weights = torch.softmax(weights, dim=1)  # 归一化

            # 加权融合（假设所有教师输出同维度）
            # 这里简化处理，实际可能需要更复杂的融合策略
            adjusted_outputs = {}
            for i, (name, output) in enumerate(teacher_outputs.items()):
                if i < num_teachers:
                    adjusted_outputs[name] = output * weights[:, i:i + 1].unsqueeze(-1).unsqueeze(-1)

            return adjusted_outputs
        else:
            return teacher_outputs
