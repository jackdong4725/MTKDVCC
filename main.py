"""
MT-FKD 主训练脚本
学生模型更新为 PointDGMamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False
    print('[main] wandb 未安装，禁用远程日志')
import sys

from config import cfg
from data.dataset import MultiDatasetLoader
from data.augmentation import EvolutionaryAugmentation as VideoMixAugmentation  # 仍可通过旧名调用
from models.scene_classifier import SceneClassifier
from models.meta_teacher import MetaTeacherGenerator
from models.experts.pointdgmamba import PointDGMambaStudent  # ✅ 使用现有实现作为学生模型
from training.teacher_selection import TeacherOrchestrator
from training.distillation_loss import DistillationLoss, compute_student_confidence
from training.difficulty_aware import DifficultyAwareModule
from evaluation.metrics import calculate_mae, calculate_rmse, calculate_mae_per_frame, calculate_information_gain

import os



def train_model(train_loader, val_loader, device, student_model, meta_teacher, scene_classifier,
                teacher_orchestrator, optimizer_student, optimizer_meta, scheduler_student,
                video_augmentor=None, difficulty_module=None,
                distillation_loss_fn=None,
                start_epoch=0, prefix="run", save_interval=None):
    """通用训练循环，可传入不同 loader 和模型配置"""
    best_mae = float('inf')
    epochs = cfg.epochs
    if save_interval is None:
        save_interval = cfg.save_interval

    for epoch in range(start_epoch, epochs):
        # ==================== 训练阶段 ====================
        student_model.train()
        if meta_teacher:
            meta_teacher.train()
        scene_classifier.eval()

        epoch_losses = {
            'total': 0, 'l1': 0, 'kl': 0,
            'temporal': 0, 'calibration': 0
        }
        epoch_metrics = {'train_mae': 0, 'train_rmse': 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            frames = batch['frames'].to(device)
            density_maps = batch['density_maps'].to(device)

            # ========== 数据增强 ==========
            if cfg.is_video_augmentation_enabled and video_augmentor is not None:
                frames, density_maps = video_augmentor(frames, density_maps)

            # ========== 场景分类 ==========
            with torch.no_grad():
                scene_probs, scene_features = scene_classifier(frames, density_maps)

            # ========== 动态教师选择 ==========
            selected_teachers, teacher_weights = teacher_orchestrator.select_teachers(
                scene_probs,
                strategy=cfg.teacher_strategy,
                top_k=cfg.teacher_top_k,
                temperature=cfg.teacher_selection_temperature
            )

            # ========== 获取教师输出 ==========
            teacher_outputs_list = teacher_orchestrator.get_teacher_outputs(
                selected_teachers,
                {'frames': frames, 'density_maps': density_maps}
            )

            # ========== 学生前向（带 domain_id）==========
            dataset_name = batch.get('dataset', 'MALL')
            domain_id = cfg.datasets.index(dataset_name) if dataset_name in cfg.datasets else 0

            if meta_teacher:
                student_output = student_model(frames, domain_id=domain_id, return_features=True)
                student_state = student_output['features']

                xi, lambda_kl = meta_teacher(scene_features, student_state)
            else:
                student_output = student_model(frames, domain_id=domain_id)
                lambda_kl = cfg.lambda_kl_initial

            # ========== 难度感知 ==========
            if cfg.is_difficulty_aware and difficulty_module and (batch_idx % cfg.difficulty_update_freq == 0):
                difficulty_scores = difficulty_module.compute_difficulty(
                    frames, teacher_outputs_list, student_output, device
                )

                if isinstance(lambda_kl, torch.Tensor):
                    lambda_kl = lambda_kl * (1 + difficulty_scores.unsqueeze(1) * cfg.difficulty_lambda_scale)
                else:
                    lambda_kl = lambda_kl * (1 + difficulty_scores.mean().item() * cfg.difficulty_lambda_scale)

            # ========== 融合教师输出 ==========
            fused_teacher_output = teacher_orchestrator.fuse_teacher_outputs(
                teacher_outputs_list, teacher_weights
            )

            # ========== 计算蒸馏损失 ==========
            student_confidence = compute_student_confidence(student_output)

            # distillation_loss_fn 应由调用者传入
            loss, loss_dict = distillation_loss_fn(
                student_output,
                fused_teacher_output,
                lambda_kl,
                epoch,
                student_confidence=student_confidence
            )

            # ========== 反向传播 ==========
            optimizer_student.zero_grad()
            if optimizer_meta:
                optimizer_meta.zero_grad()

            # detach previous graph to prevent accidental double-backward
            # wrapping in try to fallback if error persists
            try:
                loss = loss.detach().requires_grad_(True)
                loss.backward()
            except RuntimeError as e:
                print(f"[train_model] backward error, retry with retain_graph: {e}")
                loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            if meta_teacher:
                torch.nn.utils.clip_grad_norm_(meta_teacher.parameters(), 1.0)

            optimizer_student.step()
            if optimizer_meta:
                optimizer_meta.step()

            # ========== 计算训练指标 ==========
            with torch.no_grad():
                student_density = student_output['density_map']

                if student_density.shape != density_maps.shape:
                    student_density_aligned = F.interpolate(
                        student_density.view(-1, 1, *student_density.shape[2:]),
                        size=density_maps.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    ).view(density_maps.shape)
                else:
                    student_density_aligned = student_density

                train_mae = calculate_mae(student_density_aligned, density_maps)
                train_rmse = calculate_rmse(student_density_aligned, density_maps)

                epoch_metrics['train_mae'] += train_mae
                epoch_metrics['train_rmse'] += train_rmse

            for key, val in loss_dict.items():
                epoch_losses[key] += val.item() if isinstance(val, torch.Tensor) else val
            epoch_losses['total'] += loss.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mae': f"{train_mae:.2f}",
                'lr': f"{optimizer_student.param_groups[0]['lr']:.2e}"
            })

        scheduler_student.step()

        # ========== Epoch 统计 ==========
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs} 完成")
        print(f"{'='*80}")
        print(f"损失: {epoch_losses}")
        print(f"训练指标: MAE={epoch_metrics['train_mae']:.2f}, RMSE={epoch_metrics['train_rmse']:.2f}")

        if cfg.use_wandb and WANDB_AVAILABLE:
            wandb.log({**epoch_losses, **epoch_metrics, 'epoch': epoch+1})
        elif cfg.use_wandb and not WANDB_AVAILABLE:
            print('[主] cfg.use_wandb=True，但 wandb 未安装或不可用，跳过 WandB 日志')

        # ========== 评估阶段 ==========
        if (epoch + 1) % cfg.eval_interval == 0:
            print(f"\n{'=' * 80}")
            print("验证集评估")
            print(f"{'=' * 80}")

            val_mae, val_rmse, val_mae_per_frame = evaluate(
                student_model, val_loader, device
            )

            print(f"验证集 MAE: {val_mae:.2f}")
            print(f"验证集 RMSE: {val_rmse:.2f}")
            print(f"逐帧MAE: {val_mae_per_frame:.2f}")

            if cfg.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_mae_per_frame': val_mae_per_frame,
                    'epoch': epoch + 1
                })
            elif cfg.use_wandb and not WANDB_AVAILABLE:
                print('[主] cfg.use_wandb=True，但 wandb 未安装或不可用，跳过 WandB 日志')

            # best model 保存
            if val_mae < best_mae:
                best_mae = val_mae
                print(f"\n[最佳] 新的最佳 MAE: {best_mae:.2f}！保存模型...")
                save_checkpoint(
                    epoch, student_model, meta_teacher,
                    optimizer_student, optimizer_meta,
                    prefix=prefix, filename="best_model.pth"
                )

        # 保存检查点
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            save_checkpoint(epoch, student_model, meta_teacher, optimizer_student, optimizer_meta, prefix=prefix)

    if cfg.use_wandb:
        wandb.finish()

    return student_model


def main():
    print("=" * 80)
    print("MT-FKD: Meta-Teaching Framework for Adaptive Knowledge Distillation")
    print("学生模型: PointDGMamba")  # ✅ 更新
    print("专家集群: CountVid + GraspMamba + CrowdMPM + OMAN")
    print("=" * 80)

    device = torch.device(cfg.device)

    # ========== 数据加载 ==========
    print("\n[步骤 1/8] 加载数据集...")
    train_loader = MultiDatasetLoader(cfg.datasets, split='train', batch_size=cfg.batch_size)
    val_loader = MultiDatasetLoader(cfg.datasets, split='val', batch_size=cfg.batch_size)
    print(f"训练集: {len(train_loader)} batches")
    print(f"验证集: {len(val_loader)} batches")

    # ========== 场景分类器 ==========
    print("\n[步骤 2/8] 初始化场景分类器...")
    scene_classifier = SceneClassifier().to(device)
    print(f"场景类别: {cfg.num_scene_classes}")

    # ========== 元教师生成器 ==========
    print("\n[步骤 3/8] 初始化元教师生成器...")
    if cfg.is_meta_enabled:
        meta_teacher = MetaTeacherGenerator(student_feature_dim=cfg.student_embed_dim).to(device)  # ✅ 更新维度
        print(f"元教师输出维度: {cfg.meta_output_dim}")
    else:
        meta_teacher = None
        print("元教师已禁用")

    # ========== 加载专家集群 ==========
    print("\n[步骤 4/8] 加载专家集群...")
    experts = load_experts(device)
    print(f"已加载专家: {list(experts.keys())}")

    teacher_orchestrator = TeacherOrchestrator(experts)

    # ========== 学生模型 ==========
    print("\n[步骤 5/8] 初始化学生模型 (PointDGMamba)...")  # ✅ 更新
    try:
        student_model = PointDGMambaStudent().to(device)  # ✅ 更新
        from utils.helpers import count_parameters
        print(f"学生参数量: {count_parameters(student_model) / 1e6:.2f}M")
    except Exception as e:
        print(f"[错误] 学生模型加载失败: {e}")
        raise

    # ========== 数据增强 ==========
    print("\n[步骤 6/8] 初始化数据增强...")
    if cfg.is_video_augmentation_enabled:
        video_augmentor = VideoMixAugmentation()
        print(f"演化式数据增强概率: {cfg.videomix_prob}")
    else:
        video_augmentor = None

    # ========== 难度感知 ==========
    print("\n[步骤 7/8] 初始化难度感知模块...")
    if cfg.is_difficulty_aware:
        difficulty_module = DifficultyAwareModule()
        print(f"难度更新频率: 每 {cfg.difficulty_update_freq} batches")
    else:
        difficulty_module = None

    # ========== 损失与优化器 ==========
    print("\n[步骤 8/8] 初始化损失函数与优化器...")
    distillation_loss_fn = DistillationLoss()

    optimizer_student = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    if meta_teacher:
        optimizer_meta = torch.optim.Adam(
            meta_teacher.parameters(),
            lr=cfg.lr_meta
        )
    else:
        optimizer_meta = None

    scheduler_student = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_student, T_0=10, T_mult=2
    )

    # ========== WandB ==========
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=cfg.to_serializable_dict()
        )

    # ========== 训练循环 ==========
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)

    # prefix 可今后从 cfg 自定义，这里默认使用 time-stamp 或简单字符串
    prefix = getattr(cfg, 'experiment_prefix', 'run')

    # 尝试从已有 checkpoint 恢复
    start_epoch = 0
    latest_ckpt = cfg.checkpoint_dir / f"{prefix}_latest.pth"
    if getattr(cfg, 'resume', False) and latest_ckpt.exists():
        print(f"[恢复] 从最新检查点 {latest_ckpt} 继续训练")
        start_epoch = load_checkpoint(str(latest_ckpt), student_model, meta_teacher, optimizer_student, optimizer_meta)

    # 调用通用训练循环
    # loss 函数在本地创建并传入
    distillation_loss_fn = DistillationLoss()
    train_model(
        train_loader,
        val_loader,
        device,
        student_model,
        meta_teacher,
        scene_classifier,
        teacher_orchestrator,
        optimizer_student,
        optimizer_meta,
        scheduler_student,
        video_augmentor,
        difficulty_module,
        distillation_loss_fn=distillation_loss_fn,
        start_epoch=start_epoch,
        prefix=prefix,
        save_interval=cfg.save_interval
    )

    if cfg.use_wandb:
        wandb.finish()


def load_experts(device):
    """加载所有启用的专家模型"""
    experts = {}

    if cfg.is_countvid_enabled:
        try:
            from models.experts.countvid import CountVid
            experts['CountVid'] = CountVid().to(device).eval()
            print("[专家] CountVid 加载成功")
        except Exception as e:
            print(f"[警告] CountVid 加载失败: {e}")

    if cfg.is_graspmamba_enabled:
        try:
            from models.experts.graspmamba import GraspMambaExpert
            experts['GraspMamba'] = GraspMambaExpert().to(device).eval()
            print("[专家] GraspMamba (zero-shot) 加载成功")
        except Exception as e:
            print(f"[警告] GraspMamba 加载失败: {e}")

    # RefAtomNet 已移除，不再加载

    if cfg.is_crowdmpm_enabled:
        try:
            from models.experts.crowdmpm import CrowdMPM
            crowd = CrowdMPM()
            # ensure device compatibility even if torch is CPU-only
            if torch.cuda.is_available() and device.type == 'cuda':
                crowd = crowd.to(device)
            else:
                crowd = crowd.to('cpu')
            crowd.eval()
            experts['CrowdMPM'] = crowd
            print("[专家] CrowdMPM 加载成功")
        except Exception as e:
            print(f"[警告] CrowdMPM 加载失败: {e}")

    if cfg.is_oman_enabled:
        try:
            from models.experts.oman import OMAN
            experts['OMAN'] = OMAN().to(device).eval()
            print("[专家] OMAN 加载成功")
        except Exception as e:
            print(f"[警告] OMAN 加载失败: {e}")


    if len(experts) == 0:
        raise ValueError("未成功加载任何专家模型，请检查配置和依赖")

    return experts


def evaluate(model, loader, device):
    """评估函数"""
    model.eval()

    all_pred_counts = []
    all_gt_counts = []
    all_per_frame_abs_errors = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估中"):
            frames = batch['frames'].to(device)
            gt_density = batch['density_maps'].to(device)
            B, T, H_gt, W_gt = gt_density.shape

            # ✅ 带 domain_id
            dataset_name = batch.get('dataset', 'MALL')
            domain_id = cfg.datasets.index(dataset_name) if dataset_name in cfg.datasets else 0

            pred_output = model(frames, domain_id=domain_id)
            pred_density = pred_output['density_map']

            if pred_density.shape[2:] != gt_density.shape[2:]:
                pred_density = F.interpolate(
                    pred_density.view(-1, 1, *pred_density.shape[2:]),
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).view(B, T, H_gt, W_gt)

            pred_count = pred_density.sum(dim=[1, 2, 3])
            gt_count = gt_density.sum(dim=[1, 2, 3])

            all_pred_counts.append(pred_count)
            all_gt_counts.append(gt_count)

            pred_count_per_frame = pred_density.sum(dim=[2, 3])
            gt_count_per_frame = gt_density.sum(dim=[2, 3])

            all_per_frame_abs_errors.append(torch.abs(pred_count_per_frame - gt_count_per_frame))

    all_pred_counts = torch.cat(all_pred_counts)
    all_gt_counts = torch.cat(all_gt_counts)
    all_per_frame_abs_errors = torch.cat(all_per_frame_abs_errors)

    mae = torch.abs(all_pred_counts - all_gt_counts).mean().item()
    rmse = torch.sqrt(((all_pred_counts - all_gt_counts) ** 2).mean()).item()
    mae_per_frame = all_per_frame_abs_errors.mean().item()

    return mae, rmse, mae_per_frame


def save_checkpoint(epoch, student, meta, opt_student, opt_meta, prefix="run", filename=None):
    """保存检查点。

    ``prefix`` 用于给文件命名并更新最新链接，这样不同实验可以共存。
    ``filename`` 若不提供则会根据 prefix 和 epoch 自动生成。
    """
    checkpoint = {
        'epoch': epoch,
        'student_state_dict': student.state_dict(),
        'optimizer_student_state_dict': opt_student.state_dict(),
    }
    if meta:
        checkpoint['meta_state_dict'] = meta.state_dict()
        checkpoint['optimizer_meta_state_dict'] = opt_meta.state_dict()

    if filename is None:
        filename = f"{prefix}_checkpoint_epoch_{epoch + 1}.pth"

    save_path = cfg.checkpoint_dir / filename
    torch.save(checkpoint, save_path)
    print(f"\n[保存] 检查点已保存到: {save_path}")
    # 更新 latest 软链接
    latest = cfg.checkpoint_dir / f"{prefix}_latest.pth"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        os.symlink(save_path, latest)
    except Exception:
        pass
    # 保留最近若干个检查点，防止磁盘占用过大（按 prefix 分组）
    try:
        max_keep = 5
        files = sorted([p for p in cfg.checkpoint_dir.glob(f"{prefix}_checkpoint_epoch_*.pth") if p.is_file()], key=lambda x: x.stat().st_mtime)
        # 删除最旧的，保留最近 max_keep
        if len(files) > max_keep:
            to_remove = files[:-max_keep]
            for p in to_remove:
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass


def load_checkpoint(filepath, student, meta=None, opt_student=None, opt_meta=None):
    """加载检查点并返回起始 epoch"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    ckpt = torch.load(filepath, map_location='cpu')
    start_epoch = ckpt.get('epoch', 0)
    student.load_state_dict(ckpt['student_state_dict'])
    if opt_student is not None and 'optimizer_student_state_dict' in ckpt:
        opt_student.load_state_dict(ckpt['optimizer_student_state_dict'])
    if meta and 'meta_state_dict' in ckpt:
        meta.load_state_dict(ckpt['meta_state_dict'])
    if opt_meta and 'optimizer_meta_state_dict' in ckpt:
        opt_meta.load_state_dict(ckpt['optimizer_meta_state_dict'])
    return start_epoch


if __name__ == "__main__":
    from utils.helpers import set_seed
    set_seed(42)
    main()
