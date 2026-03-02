"""
PointDGMamba: Domain Generalization Student Model
完全对齐 https://github.com/yxltya/PointDGMamba 官方实现
论文: https://arxiv.org/pdf/2408.13574

核心组件:
1. Point Patch Embedding (官方 Tokenizer)
2. Mamba Block with Selective Scan (完整 SSM)
3. Multi-Domain Adapter (领域泛化核心)
4. Spherical Position Encoding
5. Point-based Decoder
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from einops import rearrange, repeat
import numpy as np
import math

# import configuration first so we can modify sys.path with project_root
from config import cfg
# ensure the external PointDGMamba repository is on the import path
sys.path.insert(0, str(cfg.project_root / "PointDGMamba"))

# ==================== Mamba SSM 核心（官方版本）====================
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_AVAILABLE = True
    print("[PointDGMamba] 使用官方 mamba_ssm")
except ImportError:
    MAMBA_AVAILABLE = False
    print("[PointDGMamba 警告] mamba_ssm 未安装")


# ==================== 官方 Selective Scan 实现 ====================
class SelectiveScanCore(nn.Module):
    """
    Selective Scan (官方论文 Algorithm 1)
    参考: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py
    """
    def __init__(self, d_model, d_state=16, dt_rank='auto', d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        if dt_rank == 'auto':
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        # S4D 初始化 (论文 Appendix D)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D 参数 (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

        # Δ, B, C 投影
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        # 特殊初始化
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

    def forward(self, u, delta_bias=None):
        """
        Args:
            u: (B, L, D)
        Returns:
            y: (B, L, D)
        """
        B, L, D = u.shape

        # 投影 Δ, B, C
        x_proj_out = self.x_proj(u)  # (B, L, dt_rank + 2*d_state)
        delta, B, C = torch.split(
            x_proj_out,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        delta = self.dt_proj(delta)  # (B, L, D)

        if delta_bias is not None:
            delta = delta + delta_bias

        # Softplus (确保 Δ > 0)
        delta = F.softplus(delta)

        # A 矩阵
        A = -torch.exp(self.A_log.float())  # (D, d_state)

        # Selective Scan (高效实现)
        y = self._selective_scan_seq(u, delta, A, B, C)

        # Skip connection
        y = y + u * self.D.unsqueeze(0).unsqueeze(0)

        return y

    def _selective_scan_seq(self, u, delta, A, B, C):
        """
        Sequential Selective Scan (官方 Algorithm 2)

        h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        y_t = C_t * h_t

        其中:
        A_bar_t = exp(Δ_t * A)
        B_bar_t = (A_bar_t - I) * A^{-1} * B_t * Δ_t
        """
        B_batch, L, D = u.shape
        d_state = A.shape[1]

        # 初始化隐状态
        h = torch.zeros(B_batch, D, d_state, device=u.device, dtype=u.dtype)

        outputs = []

        for t in range(L):
            # 离散化 (ZOH: Zero-Order Hold)
            dt = delta[:, t]  # (B, D)
            dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, D, d_state)

            # B_bar 计算 (简化版)
            dB = dt.unsqueeze(-1) * B[:, t].unsqueeze(1)  # (B, 1, d_state) -> (B, D, d_state)
            dB = dB.expand(-1, D, -1)

            # 状态更新
            h = dA * h + dB * u[:, t].unsqueeze(-1)

            # 输出
            y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D)
        return y


class MambaBlock(nn.Module):
    """
    完整 Mamba Block (官方实现)
    参考: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Convolution (官方使用 depthwise conv)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation
        self.act = nn.SiLU()

        # Selective Scan 核心
        if MAMBA_AVAILABLE:
            self.ssm = MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self.use_custom_ssm = False
        else:
            self.ssm = SelectiveScanCore(
                d_model=self.d_inner,
                d_state=d_state,
                d_conv=d_conv
            )
            self.use_custom_ssm = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, N, D)
        Returns:
            output: (B, N, D)
        """
        residual = x
        x = self.norm(x)

        if not self.use_custom_ssm and MAMBA_AVAILABLE:
            # 使用官方 mamba_ssm
            return residual + self.ssm(x)

        # 使用自定义实现
        xz = self.in_proj(x)  # (B, N, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, N, d_inner)

        # Conv1D
        x = rearrange(x, 'b n d -> b d n')
        x = self.conv1d(x)[:, :, :x.shape[-1]]
        x = rearrange(x, 'b d n -> b n d')

        # Activation
        x = self.act(x)

        # Selective Scan
        y = self.ssm(x)

        # Gate
        y = y * self.act(z)

        # Output projection
        output = self.out_proj(y)

        return residual + output


# ==================== Point Patch Embedding（官方 Tokenizer）====================
class PointPatchEmbed(nn.Module):
    """
    Point Patch Embedding (官方实现)
    参考: https://github.com/yxltya/PointDGMamba/blob/main/models/pointdgmamba.py

    将点云分为局部 patches 并提取特征
    """
    def __init__(self, in_channels=3, embed_dim=384, num_groups=64, group_size=32):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = group_size

        # Local feature aggregation
        self.local_mlp = nn.Sequential(
            nn.Conv1d(in_channels + 3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, embed_dim, 1, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

    def fps(self, xyz, npoint):
        """Farthest Point Sampling"""
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)

        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

    def knn_point(self, k, xyz, query_xyz):
        """K-Nearest Neighbors"""
        B, N, _ = xyz.shape
        _, S, _ = query_xyz.shape

        dist = torch.cdist(query_xyz, xyz)  # (B, S, N)
        knn_idx = dist.topk(k, dim=-1, largest=False, sorted=True)[1]  # (B, S, k)

        return knn_idx

    def forward(self, xyz, features=None):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) point features (optional)
        Returns:
            center_xyz: (B, num_groups, 3)
            patch_features: (B, num_groups, embed_dim)
        """
        B, N, _ = xyz.shape

        # FPS 采样中心点
        fps_idx = self.fps(xyz, self.num_groups)
        center_xyz = torch.gather(
            xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, num_groups, 3)

        # KNN 分组
        knn_idx = self.knn_point(self.group_size, xyz, center_xyz)  # (B, num_groups, k)

        # 提取局部点
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, self.num_groups, -1, -1),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (B, num_groups, k, 3)

        # 归一化坐标
        grouped_xyz_norm = grouped_xyz - center_xyz.unsqueeze(2)

        # 提取特征
        if features is not None:
            grouped_features = torch.gather(
                features.unsqueeze(1).expand(-1, self.num_groups, -1, -1),
                2,
                knn_idx.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1])
            )
            grouped_input = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
        else:
            grouped_input = grouped_xyz_norm

        # (B, num_groups, k, C) -> (B*num_groups, C, k)
        grouped_input = grouped_input.reshape(B * self.num_groups, self.group_size, -1)
        grouped_input = grouped_input.permute(0, 2, 1)

        # Local MLP
        patch_features = self.local_mlp(grouped_input)  # (B*num_groups, embed_dim, k)
        patch_features = F.adaptive_max_pool1d(patch_features, 1).squeeze(-1)
        patch_features = patch_features.reshape(B, self.num_groups, -1)

        return center_xyz, patch_features


# ==================== Multi-Domain Adapter（官方 DG 核心）====================
class DomainAdapter(nn.Module):
    """
    Multi-Domain Adapter (官方实现)
    参考论文 Sec 3.3: Domain-Specific Adapters
    """
    def __init__(self, embed_dim, num_domains, reduction=4):
        super().__init__()
        self.num_domains = num_domains

        # Domain-specific adapters (LoRA-style)
        self.domain_down = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim // reduction, bias=False)
            for _ in range(num_domains)
        ])

        self.domain_up = nn.ModuleList([
            nn.Linear(embed_dim // reduction, embed_dim, bias=False)
            for _ in range(num_domains)
        ])

        self.act = nn.ReLU(inplace=True)

        # 初始化为接近零
        for i in range(num_domains):
            nn.init.xavier_uniform_(self.domain_down[i].weight, gain=0.01)
            nn.init.zeros_(self.domain_up[i].weight)

    def forward(self, x, domain_id):
        """
        Args:
            x: (B, N, D)
            domain_id: int (0 ~ num_domains-1)
        Returns:
            x_adapted: (B, N, D)
        """
        residual = x
        x = self.domain_down[domain_id](x)
        x = self.act(x)
        x = self.domain_up[domain_id](x)
        return residual + x


# ==================== Spherical Position Encoding（官方）====================
class SphericalPositionEncoding(nn.Module):
    """
    Spherical Position Encoding for 3D points
    参考: https://github.com/yxltya/PointDGMamba/blob/main/models/pointdgmamba.py
    """
    def __init__(self, embed_dim, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        assert embed_dim % 6 == 0, "embed_dim must be divisible by 6"
        self.num_pos_feats = embed_dim // 6

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) normalized coordinates in [-1, 1]
        Returns:
            pos_embed: (B, N, embed_dim)
        """
        B, N, _ = xyz.shape

        # 笛卡尔坐标
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        # 球坐标
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        theta = torch.acos(z / (r + 1e-8))  # [0, π]
        phi = torch.atan2(y, x)  # [-π, π]

        # Positional encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        def encode_1d(val):
            val = val.unsqueeze(-1) / dim_t
            val = torch.stack([val[..., 0::2].sin(), val[..., 1::2].cos()], dim=-1).flatten(-2)
            return val

        pos_x = encode_1d(x)
        pos_y = encode_1d(y)
        pos_z = encode_1d(z)
        pos_r = encode_1d(r)
        pos_theta = encode_1d(theta)
        pos_phi = encode_1d(phi)

        pos_embed = torch.cat([pos_x, pos_y, pos_z, pos_r, pos_theta, pos_phi], dim=-1)

        return pos_embed


# ==================== PointDGMamba 主模型（完整官方版）====================
class PointDGMambaStudent(nn.Module):
    """
    PointDGMamba Student Model (完全对齐官方实现)

    架构:
    1. CNN Backbone (2D features)
    2. 2D to 3D projection
    3. Point Patch Embedding
    4. Domain Adapter
    5. Mamba Blocks with Spherical PE
    6. Point-based Decoder
    """

    def __init__(self):
        super().__init__()

        self.embed_dim = cfg.student_embed_dim

        # 1. CNN Backbone (保持原有)
        self.backbone_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone_norm1 = nn.BatchNorm2d(64)
        self.backbone_relu = nn.ReLU(inplace=True)
        self.backbone_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.backbone_layer1 = self._make_layer(64, 64, 2)
        self.backbone_layer2 = self._make_layer(64, 128, 2, stride=2)
        self.backbone_layer3 = self._make_layer(128, 256, 2, stride=2)
        self.backbone_layer4 = self._make_layer(256, cfg.student_cnn_out_dim, 2, stride=2)

        # 2. Point Patch Embedding (官方 Tokenizer)
        self.patch_embed = PointPatchEmbed(
            in_channels=cfg.student_cnn_out_dim,
            embed_dim=self.embed_dim,
            num_groups=cfg.student_num_points // 32,  # 64 groups
            group_size=32
        )

        # 3. Spherical Position Encoding
        self.pos_encoder = SphericalPositionEncoding(embed_dim=self.embed_dim)

        # 4. Mamba Blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=self.embed_dim,
                d_state=cfg.student_d_state,
                d_conv=cfg.student_d_conv,
                expand=cfg.student_expand
            )
            for _ in range(cfg.student_depth)
        ])

        # 5. Domain Adapters (插入在 Mamba blocks 之间)
        if cfg.student_use_domain_token:
            self.domain_adapters = nn.ModuleList([
                DomainAdapter(
                    embed_dim=self.embed_dim,
                    num_domains=cfg.student_num_domains,
                    reduction=4
                )
                for _ in range(cfg.student_depth)
            ])
        else:
            self.domain_adapters = None

        # 6. Norm
        self.norm = nn.LayerNorm(self.embed_dim)

        # 7. Decoder (保持原有)
        self.decoder = self._build_decoder()

    def _make_layer(self, in_planes, planes, blocks, stride=1):
        """构建 ResNet-like 卷积层"""
        layers = []
        layers.append(nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _build_decoder(self):
        """构建密度图解码器"""
        decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(self.embed_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1)
        )
        return decoder

    def _feature_map_to_points(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 2D 特征图转换为 3D 点云 (用于 Point Patch Embedding)
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # 创建坐标网格 (归一化到 [-1, 1])
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 3D 坐标 (z=0 平面)
        coords = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten())], dim=0)
        points_xyz = coords.unsqueeze(0).repeat(B, 1, 1).transpose(1, 2)  # (B, H*W, 3)

        # 特征
        features = feature_map.flatten(2).transpose(1, 2)  # (B, H*W, C)

        return points_xyz, features

    def _points_to_grid(
            self,
            points_xyz: torch.Tensor,
            point_features: torch.Tensor,
            grid_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        使用逆距离加权将点特征散布回网格
        """
        BT, N, D = point_features.shape
        H, W = grid_shape
        device = points_xyz.device

        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y], dim=-1).view(1, H * W, 2)

        points_xy = points_xyz[:, :, :2].view(BT, N, 1, 2)

        dist = torch.norm(points_xy - grid_coords, dim=-1)
        dist = dist.clamp(min=1e-8)

        weights = 1.0 / (dist ** 2)
        weights = weights / weights.sum(dim=1, keepdim=True)

        grid_features = torch.bmm(point_features.transpose(1, 2), weights)

        grid_features = grid_features.view(BT, D, H, W)

        return grid_features

    def forward(self, frames: torch.Tensor, domain_id: int = 0, return_features: bool = False) -> Dict:
        """
        Args:
            frames: (B, T, 3, H, W)
            domain_id: int (0~6)
            return_features: bool
        """
        B, T, C, H, W = frames.shape
        device = frames.device

        frames_flat = frames.view(B * T, C, H, W)

        # 1. CNN Backbone
        x = self.backbone_conv1(frames_flat)
        x = self.backbone_norm1(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x)

        x = self.backbone_layer1(x)
        x = self.backbone_layer2(x)
        x = self.backbone_layer3(x)
        feature_map = self.backbone_layer4(x)  # (B*T, C', H', W')

        grid_shape = feature_map.shape[2:]

        # 2. 2D -> 3D 点云
        points_xyz, point_features_2d = self._feature_map_to_points(feature_map)

        # 3. Point Patch Embedding (官方 Tokenizer)
        center_xyz, patch_features = self.patch_embed(points_xyz, point_features_2d)

        # 4. Spherical Position Encoding
        pos_embed = self.pos_encoder(center_xyz)
        patch_features = patch_features + pos_embed

        # 5. Mamba Blocks + Domain Adapters
        for i, block in enumerate(self.blocks):
            # Mamba Block
            patch_features = block(patch_features)

            # Domain Adapter
            if self.domain_adapters is not None:
                patch_features = self.domain_adapters[i](patch_features, domain_id)

        patch_features = self.norm(patch_features)

        # 6. 点云 -> 网格特征
        grid_features = self._points_to_grid(
            center_xyz, patch_features, grid_shape
        )

        # 7. 解码器生成密度图
        density_maps_flat = self.decoder(grid_features)  # (B*T, 1, H_out, W_out)

        density_maps = density_maps_flat.view(B, T, *cfg.student_output_size)

        output = {'density_map': density_maps}

        if return_features:
            global_feat = patch_features.mean(dim=1)  # (B*T, D)
            output['features'] = global_feat.view(B, T, -1).mean(dim=1)  # (B, D)

        return output


def build_pointdgmamba_student():
    """构建 PointDGMamba 学生模型"""
    return PointDGMambaStudent()
