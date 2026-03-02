"""
Optimal Transport Loss for OMAN (GML)
基于 Sinkhorn-Knopp 和 Hinge Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


def similarity_cost(x1, x2, gamma=10):
    """
    计算相似度代价矩阵

    Args:
        x1: (B, N, D)
        x2: (B, M, D)
        gamma: 温度参数

    Returns:
        cost: (B, N, M) - 1 - normalized_similarity
    """
    N = x1.shape[1]
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.bmm(x1, x2.transpose(1, 2)) * gamma  # (B, N, M)
    sim = sim.exp()

    sim_c = sim.sum(dim=1, keepdim=True)  # Column sum
    sim_r = sim.sum(dim=2, keepdim=True)  # Row sum

    # Sinkhorn normalization
    sim = sim / (sim_c + sim_r - sim + 1e-8)

    return 1 - sim


class GML(nn.Module):
    """
    Geometric Matching Loss (OT-based)
    """

    def __init__(self, margin=0.1):
        super().__init__()
        self.ot = SamplesLoss(backend="tensorized", cost=similarity_cost, debias=False, diameter=3)
        self.margin = margin
        self.dist = None

    def forward(self, x_pre, x_cur, dist=None):
        """
        Args:
            x_pre: tuple of (z1, y1) - 前一帧的 (matched, independent) 特征
            x_cur: tuple of (z2, y2) - 当前帧的 (matched, independent) 特征
            dist: (B, N, M) - 距离权重矩阵（可选）

        Returns:
            loss_dict: {'scon_cost': OT loss, 'hinge_cost': hinge loss}
        """
        if dist is not None:
            self.dist = dist.cuda()

        z1, y1 = x_pre  # z1: (1, N, D), y1: (1, M1, D)
        z2, y2 = x_cur  # z2: (1, N, D), y2: (1, M2, D)

        B = len(z1)
        N = z1.shape[1]
        M = z2.shape[1]
        N1 = y1.shape[1]
        M1 = y2.shape[1]

        device = z1.device
        ot_loss_list = []

        for b in range(B):
            # Concatenate matched + independent
            alpha = torch.cat([torch.ones(1, N, device=device),
                               torch.zeros(1, N1, device=device)], dim=1)  # (1, N+N1)
            beta = torch.cat([torch.ones(1, M, device=device),
                              torch.zeros(1, M1, device=device)], dim=1)  # (1, M+M1)

            f1 = torch.cat([z1[b], y1[b]], dim=0).unsqueeze(0)  # (1, N+N1, D)
            f2 = torch.cat([z2[b], y2[b]], dim=0).unsqueeze(0)  # (1, M+M1, D)

            # Compute OT loss
            loss = self.ot(alpha, f1, beta, f2)
            ot_loss_list.append(loss)

        # Hinge loss (鼓励独立点之间不相似)
        hinge_loss = torch.relu(self.margin - torch.bmm(y1, y2.transpose(1, 2))).sum()

        loss_dict = {
            'scon_cost': sum(ot_loss_list) / len(ot_loss_list),
            'hinge_cost': hinge_loss
        }

        return loss_dict
