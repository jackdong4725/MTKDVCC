"""公共实验工具集"""
from config import cfg
import torch
from data.dataset import MultiDatasetLoader
from models.scene_classifier import SceneClassifier
from models.meta_teacher import MetaTeacherGenerator
from models.experts.pointdgmamba import PointDGMambaStudent
from data.augmentation import EvolutionaryAugmentation as VideoMixAugmentation
from training.teacher_selection import TeacherOrchestrator
from training.distillation_loss import DistillationLoss
from training.difficulty_aware import DifficultyAwareModule
from main import load_experts, evaluate
import importlib.util
import os
from pathlib import Path

# Robustly load `utils/helpers.py` from project root to avoid shadowing by other `utils` modules
project_root = Path(__file__).resolve().parents[1]
helpers_path = project_root / 'utils' / 'helpers.py'
if helpers_path.exists():
    spec = importlib.util.spec_from_file_location('project_utils_helpers', str(helpers_path))
    _helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_helpers)
    set_seed = _helpers.set_seed
else:
    # fallback to normal import (will raise if unavailable)
    from utils.helpers import set_seed


def build_components(dataset_list=None,
                     sample_fraction=None,
                     use_video_aug=None,
                     use_difficulty=None):
    """构建训练所需的所有组件。

    Args:
        dataset_list (list[str], optional): 训练/验证的数据集名称列表，
            若为 None 则使用 cfg.datasets。
        sample_fraction (float, optional): 在训练时采样的比例 (0-1)，
            用于低数据实验。
        use_video_aug (bool, optional): 是否启用演化式增强，
            None 表示使用 cfg.is_video_augmentation_enabled。
        use_difficulty (bool, optional): 是否启用难度感知模块，
            None 表示使用 cfg.is_difficulty_aware。

    Returns:
        dict: 包含 device、train_loader、val_loader、模型、优化器等。
    """
    device = torch.device(cfg.device)

    if dataset_list is not None:
        cfg.datasets = dataset_list
    # 初始化数据加载
    train_loader = MultiDatasetLoader(
        cfg.datasets, split='train', batch_size=cfg.batch_size,
        shuffle=True, sample_fraction=sample_fraction
    )
    val_loader = MultiDatasetLoader(
        cfg.datasets, split='val', batch_size=cfg.batch_size,
        shuffle=False
    )

    scene_classifier = SceneClassifier().to(device)
    meta_teacher = MetaTeacherGenerator(student_feature_dim=cfg.student_embed_dim).to(device) if cfg.is_meta_enabled else None

    experts = load_experts(device)
    teacher_orchestrator = TeacherOrchestrator(experts)

    student_model = PointDGMambaStudent().to(device)

    video_aug_flag = cfg.is_video_augmentation_enabled if use_video_aug is None else use_video_aug
    video_augmentor = VideoMixAugmentation() if video_aug_flag else None

    diff_flag = cfg.is_difficulty_aware if use_difficulty is None else use_difficulty
    difficulty_module = DifficultyAwareModule() if diff_flag else None

    optimizer_student = torch.optim.AdamW(
        student_model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    optimizer_meta = None
    if meta_teacher:
        optimizer_meta = torch.optim.Adam(
            meta_teacher.parameters(),
            lr=cfg.lr_meta
        )

    scheduler_student = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_student, T_0=10, T_mult=2
    )

    loss_fn = DistillationLoss()

    return {
        'device': device,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'scene_classifier': scene_classifier,
        'meta_teacher': meta_teacher,
        'teacher_orchestrator': teacher_orchestrator,
        'student_model': student_model,
        'optimizer_student': optimizer_student,
        'optimizer_meta': optimizer_meta,
        'scheduler_student': scheduler_student,
        'video_augmentor': video_augmentor,
        'difficulty_module': difficulty_module,
        'loss_fn': loss_fn
    }


def evaluate_all_domains(model, device):
    """在 cfg.datasets 列表中的每个数据集上评估模型。

    返回一个字典，键为数据集名，值为 (mae, rmse, mae_per_frame)。
    """
    results = {}
    for ds in cfg.datasets:
        loader = MultiDatasetLoader([ds], split='val', batch_size=cfg.batch_size, shuffle=False).loader
        mae, rmse, mae_pf = evaluate(model, loader, device)
        results[ds] = (mae, rmse, mae_pf)
    return results
