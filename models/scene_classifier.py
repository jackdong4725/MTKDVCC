"""
场景分类器：7类时空场景在线感知
修复：时空稀疏度逐样本计算、方向一致性数值稳定、使用weights参数
"""
import torch
import torch.nn as nn
import torchvision.models as models
from config import cfg


class SceneClassifier(nn.Module):
    """
    场景分类器
    输入：视频帧 + 密度图 + 光流（可选）
    输出：场景概率分布 + 全局特征
    """

    def __init__(self):
        super().__init__()

        # 使用 weights 参数（torchvision 0.13+）
        try:
            from torchvision.models import ResNet18_Weights
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            resnet = models.resnet18(pretrained=True)

        # 冻结特征提取器
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 降维层（512 -> 128）
        self.feature_projection = nn.Linear(512, 128)

        # MLP 分类器 (135D -> 7)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.meta_input_dim, cfg.scene_classifier_hidden[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.scene_classifier_hidden[0], cfg.scene_classifier_hidden[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.scene_classifier_hidden[1], cfg.num_scene_classes)
        )

    def extract_spatiotemporal_features(self, frames, density_maps=None, optical_flow=None):
        """
        提取7个时空特征（逐样本计算）

        Args:
            frames: (B, T, C, H, W)
            density_maps: (B, T, H, W)
            optical_flow: (B, T-1, 2, H, W)
        Returns:
            features: (B, 7)
        """
        B, T = frames.shape[:2]
        device = frames.device
        features = []

        if density_maps is not None:
            # 1. 均值密度 (逐样本)
            mean_density = density_maps.mean(dim=[1, 2, 3])  # (B,)
            features.append(mean_density.unsqueeze(1))

            # 2. 密度方差 (逐样本)
            density_var = density_maps.var(dim=[1, 2, 3])
            features.append(density_var.unsqueeze(1))

            # 3. 稀疏度 (逐样本)
            sparsity = (density_maps < 0.01).float().mean(dim=[1, 2, 3])
            features.append(sparsity.unsqueeze(1))

            # 5. 密度时序方差 (逐样本)
            density_temporal_var = density_maps.mean(dim=[2, 3]).var(dim=1)
            features.append(density_temporal_var.unsqueeze(1))

            # 7. 时空稀疏度 (修复：逐样本计算)
            spatiotemporal_sparsity = (density_maps < 0.01).float().mean(dim=[1, 2, 3])  # (B,)
            features.append(spatiotemporal_sparsity.unsqueeze(1))
        else:
            features.extend([torch.zeros(B, 1, device=device) for _ in range(5)])

        if optical_flow is not None:
            # 4. 平均光流强度 (逐样本)
            flow_magnitude = torch.sqrt(optical_flow[:, :, 0] ** 2 + optical_flow[:, :, 1] ** 2 + 1e-8)
            mean_flow = flow_magnitude.mean(dim=[1, 2, 3])
            features.append(mean_flow.unsqueeze(1))

            # 6. 方向一致性 (修复：数值稳定)
            flow_normalized = optical_flow / (flow_magnitude.unsqueeze(2) + 1e-8)
            # 相邻帧余弦相似度
            if T > 1:
                cos_sim = (flow_normalized[:, :-1] * flow_normalized[:, 1:]).sum(dim=2)
                direction_consistency = cos_sim.mean(dim=[1, 2, 3])
            else:
                direction_consistency = torch.zeros(B, device=device)
            features.append(direction_consistency.unsqueeze(1))
        else:
            features.extend([torch.zeros(B, 1, device=device) for _ in range(2)])

        return torch.cat(features, dim=1)  # (B, 7)

    def forward(self, frames, density_maps=None, optical_flow=None):
        """
        Args:
            frames: (B, T, 3, H, W)
            density_maps: (B, T, H, W)
            optical_flow: (B, T-1, 2, H, W)
        Returns:
            scene_probs: (B, 7)
            scene_features: (B, 135) - 用于 Meta-Teacher 输入
        """
        B, T, C, H, W = frames.shape

        # 提取全局视觉特征
        frames_flat = frames.view(B * T, C, H, W)
        with torch.no_grad():
            visual_features = self.feature_extractor(frames_flat)  # (B*T, 512, 1, 1)
        visual_features = visual_features.view(B, T, 512).mean(dim=1)  # (B, 512)

        # 降维到 128D
        global_features = self.feature_projection(visual_features)  # (B, 128)

        # 提取时空特征
        spatiotemporal_features = self.extract_spatiotemporal_features(
            frames, density_maps, optical_flow
        )  # (B, 7)

        # 拼接
        combined_features = torch.cat([global_features, spatiotemporal_features], dim=1)  # (B, 135)

        # 分类
        logits = self.classifier(combined_features)
        scene_probs = torch.softmax(logits, dim=1)

        return scene_probs, combined_features