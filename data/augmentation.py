"""
视频数据增强模块：通过演化式混合（Evolutionary Mix）实现数据增强
同时支持 VideoMix 和 CutMix 变体，作为论文中描述的 evolutionary augmentation 实现。
"""
import torch
import random
import numpy as np
from config import cfg


class EvolutionaryAugmentation:
    """演化式视频增强（Evolutionary Augmentation）

    该类实现了 VideoMix/CutMix 的随机组合，以模拟论文中的演化数据增强机制。
    """

    def __init__(self):
        self.prob = cfg.videomix_prob
        self.alpha = cfg.videomix_alpha

    def __call__(self, frames, density_maps):
        """
        Args:
            frames: (B, T, 3, H, W)
            density_maps: (B, T, H, W)
        Returns:
            mixed_frames, mixed_density_maps
        """
        B, T, C, H, W = frames.shape

        if random.random() > self.prob:
            return frames, density_maps

        # 生成混合权重
        lam = np.random.beta(self.alpha, self.alpha)

        # 随机配对
        indices = torch.randperm(B)

        # VideoMix（全图混合）或 CutMix（区域混合）
        if random.random() > 0.5:
            # VideoMix
            mixed_frames = lam * frames + (1 - lam) * frames[indices]
            mixed_density = lam * density_maps + (1 - lam) * density_maps[indices]
        else:
            # CutMix
            bbx1, bby1, bbx2, bby2 = self.rand_bbox((H, W), lam)

            mixed_frames = frames.clone()
            mixed_frames[:, :, :, bby1:bby2, bbx1:bbx2] = frames[indices, :, :, bby1:bby2, bbx1:bbx2]

            mixed_density = density_maps.clone()
            mixed_density[:, :, bby1:bby2, bbx1:bbx2] = density_maps[indices, :, bby1:bby2, bbx1:bbx2]

        return mixed_frames, mixed_density


    def rand_bbox(self, size, lam):
        """生成随机裁剪框"""
        H, W = size
        cut_rat = np.sqrt(1. - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


# 兼容旧名称
VideoMixAugmentation = EvolutionaryAugmentation