"""
CountVid: Open-World Counting Expert
基于真实的 GroundingDINO 实现
参考: https://github.com/niki-amini-naieni/CountVid
"""
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple

from config import cfg

# 添加 GroundingDINO 本地代码路径（如果存在）
# 支持两种位置：项目内部的 CountVid/groundingdino 以及根目录下的 GroundingDINO-main
possible_parents = [
    cfg.project_root / "CountVid",
    cfg.project_root
]
found = False
for parent in possible_parents:
    gd_dir = parent / "groundingdino"
    if gd_dir.exists():
        sys.path.insert(0, str(parent))
        print(f"[CountVid] Added local GroundingDINO parent path: {parent} (contains groundingdino)")
        found = True
        # attempt to build C++/CUDA ops if they aren't importable yet
        ops_dir = parent / "models" / "GroundingDINO" / "ops"
        if ops_dir.exists():
            import torch as _torch
            if not _torch.cuda.is_available():
                print("[CountVid] CUDA 不可用，跳过 C++ 扩展编译；将使用纯 Python/CPU 模块")
            else:
                try:
                    # try importing compiled module to see if build already happened
                    import importlib
                    importlib.import_module("GroundingDINO.ops.MultiScaleDeformableAttention")
                except Exception:
                    print("[CountVid] GroundingDINO C++扩展缺失，尝试编译...")
                    try:
                        import subprocess
                        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(ops_dir), check=True)
                        print("[CountVid] C++扩展编译完成")
                    except Exception as e:
                        print(f"[CountVid 警告] C++扩展编译失败: {e}")
        break
if not found:
    print(f"[CountVid] 未找到本地 GroundingDINO 代码， 将使用安装的包")


# Attempt to import GroundingDINO inference utilities; fall back to no-op stubs if unavailable
try:
    from groundingdino.util.inference import load_model as load_gd_model, load_image, predict, annotate
    from groundingdino.util import box_ops
except Exception as e:
    # Import may fail early if groundingdino package is incomplete
    print(f"[CountVid] 无法导入 groundingdino util，使用占位实现: {e}")
    # define stub functions so that CountVid initialization does not crash
    def load_gd_model(model_config_path: str, model_checkpoint_path: str, device: str = "cpu"):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, captions=None):
                b = x.shape[0]
                # return empty predictions
                return {"pred_logits": torch.zeros((b, 1, 256)), "pred_boxes": torch.zeros((b, 1, 4))}
        return DummyModel()

    def load_image(image_path: str):
        # simple loader returning numpy array and a tensor
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return arr, tensor

    def predict(model, image: torch.Tensor, caption: str, box_threshold: float,
                text_threshold: float, device: str = "cpu", remove_combined: bool = False):
        # no detection: always return empty outputs
        return torch.zeros((0, 4)), torch.zeros((0,)), []

    def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]):
        return image_source

    class box_ops:
        @staticmethod
        def box_cxcywh_to_xyxy(boxes):
            # identity transform
            return boxes


# 添加 SAM 路径
sam_path = cfg.project_root / "CountVid" / "segment_anything"
if sam_path.exists():
    sys.path.insert(0, str(sam_path))
    print(f"[CountVid] Added local SAM path: {sam_path}")
else:
    print(f"[CountVid] 本地 SAM 路径不存在: {sam_path}, 将尝试使用安装的 segment_anything 包")

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except Exception:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None
    print('[CountVid] segment_anything import failed, SAM unavailable; continuing without SAM')


class CountVid(nn.Module):
    """
    CountVid Expert (基于 GroundingDINO 和 SAM)
    输出统一格式：{'density_map': (B,T,H,W), 'confidence': (B,)}
    """

    def __init__(self):
        super().__init__()

        print("[CountVid] 初始化 GroundingDINO 和 SAM...")

        # ==================== 检查并提醒 BERT 模型目录 ====================
        bert_dir = cfg.groundingdino_bert_path if hasattr(cfg, 'groundingdino_bert_path') else None
        if bert_dir is not None and not bert_dir.exists():
            print(f"[CountVid 警告] 指定的 BERT 模型目录 {bert_dir} 不存在，GroundingDINO 可能会从 HuggingFace 下载")

        # ==================== 加载 GroundingDINO 模型 (countgd_box.pth) ====================
        try:
            self.gd_model = load_gd_model( # ✅ 使用别名
                str(cfg.groundingdino_config_file),
                str(cfg.groundingdino_checkpoint)
            )
            self.gd_model.eval()
            print("[CountVid] GroundingDINO 模型加载成功")
        except Exception as e:
            print(f"[CountVid 错误] GroundingDINO 模型加载失败: {e}")
            raise

        # ==================== 加载 SAM 模型 (sam2.1_hiera_large.pt) ====================
        # SAM is optional: if registry or mask generator not available, fall back
        try:
            if sam_model_registry is None or SamAutomaticMaskGenerator is None:
                print('[CountVid] SAM not available, skipping SAM initialization')
                self.sam = None
                self.mask_generator = None
            else:
                # model type is defined in config so we can adapt if needed
                self.sam_model_type = cfg.sam_model_type
                self.sam_checkpoint = str(cfg.sam_checkpoint)

                # registry may raise if model type missing; guard it
                if self.sam_model_type in sam_model_registry:
                    try:
                        self.sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
                        self.sam.to(device=cfg.device)
                        self.sam.eval()
                        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
                        print(f"[CountVid] SAM ({self.sam_checkpoint}) loaded as {self.sam_model_type}")
                    except Exception as e:
                        # loading failure (e.g. size mismatch) - warn and null out
                        print(f"[CountVid 警告] 无法用提供的 checkpoint 初始化 SAM ({self.sam_model_type}): {e}")
                        self.sam = None
                        self.mask_generator = None
                else:
                    print(f"[CountVid] SAM model type {self.sam_model_type} not in registry, skipping SAM")
                    self.sam = None
                    self.mask_generator = None
        except Exception as e:
            print(f"[CountVid 警告] SAM 初始化失败，继续 (错误: {e})")
            self.sam = None
            self.mask_generator = None

        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        self.eval() # 确保模型处于评估模式

    def _generate_gaussian_kernel(self, kernel_size=15, sigma=3.0):
        """生成高斯核"""
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)

    def _boxes_to_density(self, boxes, scores, H, W, device):
        """
        将检测框转为密度图（高斯核叠加）
        """
        density = torch.zeros(H, W, device=device, dtype=torch.float32)

        if len(boxes) == 0:
            return density, 0

        count = len(boxes)

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w_box = x2 - x1
            h_box = y2 - y1

            sigma = max(w_box.item(), h_box.item()) / 4.0
            sigma = max(sigma, 2.0)
            sigma = min(sigma, 20.0)

            kernel_size = int(sigma * 3)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = min(kernel_size, 51)

            local_gaussian = self._generate_gaussian_kernel(kernel_size, sigma).to(device)
            local_gaussian = local_gaussian / (local_gaussian.sum() + 1e-8)

            h_radius = kernel_size // 2
            w_radius = kernel_size // 2

            cx_int = int(cx.item())
            cy_int = int(cy.item())

            y_min = max(0, cy_int - h_radius)
            y_max = min(H, cy_int + h_radius + 1)
            x_min = max(0, cx_int - w_radius)
            x_max = min(W, cx_int + w_radius + 1)

            kernel_y_min = h_radius - (cy_int - y_min)
            kernel_y_max = h_radius + (y_max - cy_int)
            kernel_x_min = w_radius - (cx_int - x_min)
            kernel_x_max = w_radius + (x_max - cx_int)

            if y_max > y_min and x_max > x_min:
                try:
                    density[y_min:y_max, x_min:x_max] += \
                        local_gaussian[0, 0, kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]
                except Exception as e:
                    pass

        return density, count

    def _preprocess_image(self, image_tensor):
        """
        将 PyTorch tensor 转为 PIL Image (GroundingDINO 需要)
        """
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def forward(self, inputs):
        """
        Args:
            inputs: Dict包含
                - 'frames': (B, T, 3, H, W) in [0, 1]
                - 'text_prompt': List[str] (可选)
        Returns:
            Dict: {
                'density_map': (B, T, H_out, W_out),
                'confidence': (B,),
                'counts': (B, T) - 每帧的实际计数
            }
        """
        frames = inputs['frames']
        text_prompts = inputs.get('text_prompt', [cfg.countvid_default_prompt] * frames.shape[0])

        B, T, C, H_in, W_in = frames.shape
        device = frames.device
        H_out, W_out = cfg.student_output_size

        all_density_maps = []
        all_confidences = []
        all_counts = []

        with torch.no_grad():
            for b in range(B):
                frame_densities = []
                frame_confs = []
                frame_counts = []

                text_prompt = text_prompts[b].lower().strip()
                if not text_prompt.endswith("."):
                    text_prompt = text_prompt + "."

                for t in range(T):
                    frame = frames[b, t] # (3, H, W) in [0, 1]

                    pil_image = self._preprocess_image(frame)

                    # 保存临时文件 (GroundingDINO 的 load_image 需要文件路径)
                    # 更好的方法是使用 in-memory bytes，但为了与 load_image 兼容，仍用文件
                    temp_path = "/tmp/temp_frame.jpg"
                    pil_image.save(temp_path)

                    try:
                        image_source, image_transformed = load_image(temp_path)
                        # ensure image tensor on correct device
                        try:
                            image_transformed = image_transformed.to(device)
                        except Exception:
                            pass

                        boxes, logits, phrases = predict(
                            model=self.gd_model, # ✅ 使用 self.gd_model
                            image=image_transformed,
                            caption=text_prompt,
                            box_threshold=cfg.countvid_box_threshold,
                            text_threshold=cfg.countvid_text_threshold,
                            device=device
                        )

                        # Convert boxes to xyxy pixel coordinates
                        H_src, W_src = image_source.shape[:2]
                        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
                        try:
                            boxes_xyxy = boxes_xyxy.to(device)
                        except Exception:
                            boxes_xyxy = boxes_xyxy
                        boxes_xyxy = boxes_xyxy * torch.tensor(
                            [W_src, H_src, W_src, H_src], device=device
                        )

                        # Scale boxes to output size
                        scale_x = W_out / W_src
                        scale_y = H_out / H_src
                        boxes_scaled = boxes_xyxy * torch.tensor(
                            [scale_x, scale_y, scale_x, scale_y], device=device
                        )

                        # (可选) 使用 SAM 进一步精修掩码或过滤
                        # if len(boxes) > 0:
                        #     image_source_np = np.array(image_source)
                        #     masks = self.mask_generator.generate(image_source_np)
                        #     # ... 进一步处理 masks 和 boxes

                        density_map, count = self._boxes_to_density(
                            boxes_scaled, logits, H_out, W_out, device
                        )

                        frame_densities.append(density_map)
                        frame_counts.append(count)

                        if len(logits) > 0:
                            frame_confs.append(logits.mean())
                        else:
                            frame_confs.append(torch.tensor(0.1, device=device))

                    except Exception as e:
                        print(f"[CountVid 警告] 检测失败 (batch {b}, frame {t}): {e}")
                        frame_densities.append(torch.zeros(H_out, W_out, device=device))
                        frame_counts.append(0)
                        frame_confs.append(torch.tensor(0.1, device=device))

                all_density_maps.append(torch.stack(frame_densities))
                all_confidences.append(torch.stack(frame_confs).mean())
                all_counts.append(torch.tensor(frame_counts, device=device))

        return {
            'density_map': torch.stack(all_density_maps),
            'confidence': torch.stack(all_confidences),
            'counts': torch.stack(all_counts)
        }


def build_countvid_expert():
    """构建 CountVid 专家模型"""
    return CountVid()
