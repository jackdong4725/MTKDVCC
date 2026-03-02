"""
评估指标模块
MAE/RMSE 计算与主流人群计数论文一致
"""
import torch
import numpy as np
from config import cfg


def calculate_mae(pred_density, gt_density):
    """
    Mean Absolute Error (符合人群计数标准)

    计算方式：对每张图/视频的总计数求MAE
    MAE = mean(|sum(pred_density) - sum(gt_density)|)

    Args:
        pred_density: (B, T, H, W) 或 (B, H, W)
        gt_density: (B, T, H, W) 或 (B, H, W)
    Returns:
        float: MAE
    """
    # 对空间维度求和，得到每帧/每图的总计数
    if pred_density.dim() == 4:  # 视频 (B, T, H, W)
        # 逐帧计数
        pred_count = pred_density.sum(dim=[2, 3])  # (B, T)
        gt_count = gt_density.sum(dim=[2, 3])  # (B, T)
        # 对时间维度也求和（整个视频的总计数）
        pred_count = pred_count.sum(dim=1)  # (B,)
        gt_count = gt_count.sum(dim=1)  # (B,)
    else:  # 图像 (B, H, W)
        pred_count = pred_density.sum(dim=[1, 2])  # (B,)
        gt_count = gt_density.sum(dim=[1, 2])  # (B,)

    # MAE
    mae = torch.abs(pred_count - gt_count).mean().item()
    return mae


def calculate_rmse(pred_density, gt_density):
    """
    Root Mean Squared Error (符合人群计数标准)

    RMSE = sqrt(mean((sum(pred_density) - sum(gt_density))^2))

    Args:
        pred_density: (B, T, H, W) 或 (B, H, W)
        gt_density: (B, T, H, W) 或 (B, H, W)
    Returns:
        float: RMSE
    """
    # 对空间维度求和
    if pred_density.dim() == 4:
        pred_count = pred_density.sum(dim=[2, 3]).sum(dim=1)  # (B,)
        gt_count = gt_density.sum(dim=[2, 3]).sum(dim=1)  # (B,)
    else:
        pred_count = pred_density.sum(dim=[1, 2])  # (B,)
        gt_count = gt_density.sum(dim=[1, 2])  # (B,)

    # RMSE
    mse = ((pred_count - gt_count) ** 2).mean()
    rmse = torch.sqrt(mse).item()
    return rmse


def calculate_mae_per_frame(pred_density, gt_density):
    """
    逐帧 MAE（用于视频分析）

    Args:
        pred_density: (B, T, H, W)
        gt_density: (B, T, H, W)
    Returns:
        float: 逐帧 MAE 的平均
    """
    assert pred_density.dim() == 4, "需要视频格式 (B, T, H, W)"

    # 逐帧计数
    pred_count = pred_density.sum(dim=[2, 3])  # (B, T)
    gt_count = gt_density.sum(dim=[2, 3])  # (B, T)

    # 逐帧MAE
    mae_per_frame = torch.abs(pred_count - gt_count).mean().item()
    return mae_per_frame


def calculate_information_gain(student_model, orchestrator, scene_classifier,
                               val_loader, device):
    """
    信息增益：动态门控 vs 随机门控的 Δ验证MAE

    Returns:
        float: MAE_random - MAE_dynamic (正值表示动态门控更优)
    """
    student_model.eval()
    scene_classifier.eval()

    mae_dynamic_list = []
    mae_random_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= cfg.info_gain_eval_samples // val_loader.batch_size:
                break

            frames = batch['frames'].to(device)
            gt_density = batch['density_maps'].to(device)

            # 场景分类
            scene_probs, _ = scene_classifier(frames, gt_density)

            # 1. 动态门控
            selected_teachers, teacher_weights = orchestrator.select_teachers(
                scene_probs, strategy='soft', top_k=cfg.teacher_top_k
            )
            teacher_outputs = orchestrator.get_teacher_outputs(
                selected_teachers, {'frames': frames}
            )
            fused_dynamic = orchestrator.fuse_teacher_outputs(teacher_outputs, teacher_weights)

            if fused_dynamic is not None:
                # 对齐尺寸
                teacher_density = F.interpolate(
                    fused_dynamic['density_map'].view(-1, 1, *fused_dynamic['density_map'].shape[2:]),
                    size=gt_density.shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).view(gt_density.shape)

                mae_dynamic_list.append(calculate_mae(teacher_density, gt_density))

            # 2. 随机门控
            random_probs = torch.rand_like(scene_probs)
            random_probs = random_probs / random_probs.sum(dim=1, keepdim=True)

            selected_teachers_rand, teacher_weights_rand = orchestrator.select_teachers(
                random_probs, strategy='soft', top_k=cfg.teacher_top_k
            )
            teacher_outputs_rand = orchestrator.get_teacher_outputs(
                selected_teachers_rand, {'frames': frames}
            )
            fused_random = orchestrator.fuse_teacher_outputs(teacher_outputs_rand, teacher_weights_rand)

            if fused_random is not None:
                teacher_density_rand = F.interpolate(
                    fused_random['density_map'].view(-1, 1, *fused_random['density_map'].shape[2:]),
                    size=gt_density.shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).view(gt_density.shape)

                mae_random_list.append(calculate_mae(teacher_density_rand, gt_density))

    if len(mae_dynamic_list) == 0 or len(mae_random_list) == 0:
        return 0.0

    # 信息增益 = 随机策略误差 - 动态策略误差
    info_gain = np.mean(mae_random_list) - np.mean(mae_dynamic_list)
    return info_gain


def calculate_game(pred_density, gt_density, grid_size=4):
    """
    Grid Average Mean Absolute Error (GAME)
    常用于人群计数的局部误差评估

    Args:
        pred_density: (B, H, W)
        gt_density: (B, H, W)
        grid_size: int, 将图像分为 grid_size x grid_size 的网格
    Returns:
        float: GAME score
    """
    B, H, W = pred_density.shape
    grid_h = H // grid_size
    grid_w = W // grid_size

    game = 0.0
    count = 0

    for i in range(grid_size):
        for j in range(grid_size):
            h_start = i * grid_h
            h_end = (i + 1) * grid_h if i < grid_size - 1 else H
            w_start = j * grid_w
            w_end = (j + 1) * grid_w if j < grid_size - 1 else W

            pred_grid = pred_density[:, h_start:h_end, w_start:w_end].sum(dim=[1, 2])
            gt_grid = gt_density[:, h_start:h_end, w_start:w_end].sum(dim=[1, 2])

            game += torch.abs(pred_grid - gt_grid).sum().item()
            count += B

    return game / count if count > 0 else 0.0