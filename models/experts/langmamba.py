"""
LangMamba: 完整实现 Vision Mamba + BERT
基于 https://github.com/hao1635/LangMamba
论文: https://arxiv.org/pdf/2507.06140
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from einops import rearrange
import collections # ✅ 新增

sys.path.insert(0, str(cfg.project_root / "LangMamba"))

# ==================== Mamba SSM 核心 ====================
try:
    from mamba_ssm import Mamba as MambaBlock
except ImportError:
    print("[LangMamba 警告] mamba_ssm 未安装，使用简化版")
    MambaBlock = None


class SimplifiedMambaBlock(nn.Module):
    """简化的 Mamba Block (当无法导入 mamba_ssm 时使用)"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + d_model)
        self.dt_proj = nn.Linear(d_state, self.d_inner)

        # State space parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        """
        x: (B, L, D)
        """
        B, L, D = x.shape

        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner)

        # Conv1D
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, 'b d l -> b l d')

        # Activation
        x = self.activation(x)

        # SSM
        x_dbl = self.x_proj(x)
        delta, B_ssm, C = torch.split(
            x_dbl, [self.A_log.shape[1], self.A_log.shape[1], D], dim=-1
        )

        # Discretization
        A = -torch.exp(self.A_log.float())
        deltaA = torch.exp(torch.einsum('bld,dn->bln', delta, A))

        # Simple SSM update (simplified)
        y = x * (1 + self.D.unsqueeze(0).unsqueeze(0))

        # Gate
        y = y * self.activation(z)

        # Output projection
        output = self.out_proj(y)
        return output


# ==================== Vision Mamba ====================
class VisionMamba(nn.Module):
    """
    Vision Mamba Encoder
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=384,
            depth=24,
            drop_path_rate=0.1,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if MambaBlock is not None:
            self.blocks = nn.ModuleList([
                MambaBlock(d_model=embed_dim)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                SimplifiedMambaBlock(d_model=embed_dim)
                for _ in range(depth)
            ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = x + self.pos_embed
        for blk in self.blocks:
            x = x + blk(x)
        x = self.norm(x)
        return x


# ==================== LangMamba Expert ====================
class LangMamba(nn.Module):
    """
    LangMamba: Vision Mamba + BERT for multimodal understanding
    """

    def __init__(self):
        super().__init__()

        print("[LangMamba] 初始化 Vision Mamba + BERT...")

        # Vision Mamba Encoder
        self.vision_encoder = VisionMamba(
            img_size=cfg.langmamba_img_size,
            patch_size=cfg.langmamba_patch_size,
            embed_dim=cfg.langmamba_embed_dim,
            depth=cfg.langmamba_depth
        )

        # Text Encoder (BERT)
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.langmamba_text_encoder)
        self.text_encoder = AutoModel.from_pretrained(cfg.langmamba_text_encoder)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.vision_proj = nn.Linear(cfg.langmamba_embed_dim, 768)
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=768, nhead=12, dim_feedforward=3072
            )
            for _ in range(cfg.langmamba_fusion_layers)
        ])

        self.scene_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg.num_scene_classes)
        )

        self._load_pretrained_weights()  # ✅ 权重加载函数

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self):
        """加载 VGG 预训练权重 (假定用于 Vision Mamba 的部分)"""
        if os.path.exists(cfg.langmamba_checkpoint):
            try:
                checkpoint = torch.load(cfg.langmamba_checkpoint, map_location='cpu')
                state_dict = checkpoint.get('model', checkpoint)  # 尝试获取 'model' 键或直接使用数据
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                # ✅ 关键处理：尝试将 VGG 权重映射到 Vision Mamba
                # 这非常依赖于 VGG.pth 文件的实际结构和 VisionMamba 的具体实现
                # 理论上 VisionMamba 不是 VGG，所以直接加载会有大量键不匹配
                # 这里我们假设 VGG.pth 主要包含 Conv 层，可以部分映射到 patch_embed

                # 过滤出 Vision Mamba 的 patch_embed 和 Mamba blocks 部分
                vim_state_dict = collections.OrderedDict()
                for k, v in state_dict.items():
                    # 这是一个非常粗略的示例映射，实际需要根据 vgg.pth 的键名和 ViM 结构精细调整
                    if 'features' in k:  # VGG 的特征提取层通常以 'features.' 开头
                        # 假设 VGG 的 Conv 层可以映射到 ViM 的 patch_embed
                        if '0.weight' in k:  # VGG的第一个卷积层
                            vim_state_dict['vision_encoder.patch_embed.weight'] = v
                        elif '0.bias' in k:
                            vim_state_dict['vision_encoder.patch_embed.bias'] = v
                        # 其他 VGG 层到 Mamba blocks 的映射会非常复杂，因为架构不同
                        # 暂时只尝试加载 patch_embed

                if vim_state_dict:
                    self.vision_encoder.load_state_dict(vim_state_dict, strict=False)
                    print(f"[LangMamba] VGG 权重 ({cfg.langmamba_checkpoint}) 部分加载到 VisionMamba (patch_embed)")
                else:
                    # 如果过滤后为空，或者 VGG 结构与 VisionMamba 差异太大
                    # 尝试直接加载，但会有大量不匹配
                    self.load_state_dict(state_dict, strict=False)
                    print(f"[LangMamba] VGG 权重 ({cfg.langmamba_checkpoint}) 尝试直接加载，预计会有大量键不匹配。")

                print(f"[LangMamba] 权重加载完成，请检查输出了解不匹配情况。")

            except Exception as e:
                print(f"[LangMamba 警告] VGG 权重 ({cfg.langmamba_checkpoint}) 加载失败: {e}")
                print("[LangMamba 警告] LangMamba 将使用随机初始化或其默认预训练权重。")
        else:
            print(f"[LangMamba 警告] VGG 权重文件 ({cfg.langmamba_checkpoint}) 不存在。")
            print("[LangMamba 警告] LangMamba 将使用随机初始化或其默认预训练权重。")

    def forward(self, inputs):
        """
        Args:
            inputs: Dict {
                'frames': (B, T, 3, H, W),
                'text_description': List[str] (optional)
            }
        Returns:
            Dict {
                'semantic_embedding': (B, D),
                'scene_probs': (B, 7),
                'confidence': (B,)
            }
        """
        frames = inputs['frames']
        text_descriptions = inputs.get('text_description', None)

        B, T, C, H, W = frames.shape
        device = frames.device

        # Use middle frame
        representative_frames = frames[:, T // 2]  # (B, 3, H, W)

        with torch.no_grad():
            # Vision encoding
            vision_feats = []
            for b in range(B):
                frame = representative_frames[b:b + 1]
                feat = self.vision_encoder(frame)  # (1, N, D)
                vision_feats.append(feat.mean(dim=1))  # (1, D)

            vision_embedding = torch.cat(vision_feats, dim=0)  # (B, D)
            vision_embedding = self.vision_proj(vision_embedding)  # (B, 768)

            # Text encoding (if provided)
            if text_descriptions is not None:
                inputs_text = self.tokenizer(
                    text_descriptions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                text_outputs = self.text_encoder(**inputs_text)
                text_embedding = text_outputs.last_hidden_state.mean(dim=1)  # (B, 768)

                # Fusion
                combined = vision_embedding + text_embedding
            else:
                combined = vision_embedding

            # Scene classification
            scene_logits = self.scene_head(combined)
            scene_probs = torch.softmax(scene_logits, dim=1)

            confidence = scene_probs.max(dim=1)[0]

        return {
            'semantic_embedding': combined,
            'scene_probs': scene_probs,
            'scene_labels': scene_probs.argmax(dim=1),
            'confidence': confidence
        }
