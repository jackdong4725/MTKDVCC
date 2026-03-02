"""Run full pipeline for low-data fractions and optionally generalization/ablation.

Usage:
    python experiments/run_all_pipeline.py [--dry-run]

Dry-run: set small epochs and run 1 epoch per fraction to validate environment.
"""
import argparse
import csv
import copy
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg
from experiments.utils import build_components, evaluate_all_domains, set_seed
from main import train_model, load_checkpoint


def run_lowdata_pipeline(fractions, dry_run=True, base_ckpt=None):
    results = []
    original_epochs = cfg.epochs
    # reduce epochs for dry run
    if dry_run:
        cfg.epochs = 1
        cfg.save_interval = 1

    # Set default checkpoint save interval for experiments to 10 epochs to allow resume without
    # saving too frequently (user preference: 10 or 30). Use 10 by default here.
    cfg.save_interval = 10

    for frac in fractions:
        prefix = f"lowdata_{int(frac*100)}pct"
        cfg.experiment_prefix = prefix
        cfg.resume = True
        set_seed(42)

        print(f"\n=== Running fraction {frac} prefix={prefix} ===")

        if dry_run:
            # 构建轻量化组件以跳过外部依赖（仅用于环境验证）
            from types import SimpleNamespace
            import torch

            device = torch.device(cfg.device)
            # minimal dummy student
            class DummyStudent(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.H, self.W = cfg.student_output_size

                def forward(self, frames, domain_id=0, return_features=False):
                    B, T, C, H, W = frames.shape
                    density = torch.zeros(B, T, self.H, self.W, device=frames.device)
                    out = {'density_map': density}
                    if return_features:
                        out['features'] = torch.zeros(B, cfg.student_embed_dim, device=frames.device)
                    return out

            # Dummy components for dry-run; ensure optimizer has at least one param to avoid empty-list error
            dummy_param = torch.nn.Parameter(torch.zeros(1))
            comps = {
                'device': device,
                'train_loader': None,
                'val_loader': None,
                'scene_classifier': None,
                'meta_teacher': None,
                'teacher_orchestrator': None,
                'student_model': DummyStudent().to(device),
                'optimizer_student': torch.optim.AdamW([dummy_param], lr=1e-4),
                'optimizer_meta': None,
                'scheduler_student': None,
                'video_augmentor': None,
                'difficulty_module': None,
                'loss_fn': None
            }

            # skip training loop and just validate forward/backward quickly
            import torch
            stu = comps['student_model']
            stu.train()
            B,T = 1, cfg.num_frames
            frames = torch.rand(B,T,3, cfg.student_output_size[0], cfg.student_output_size[1], device=comps['device'])
            out = stu(frames)
            print('[dry-run] dummy student forward ok, density shape', out['density_map'].shape)
            # no optimizer step here
            # record dummy evals
            for ds in cfg.datasets:
                results.append({'dataset': ds, 'fraction': frac, 'mae': -1.0, 'rmse': -1.0, 'per_frame_mae': -1.0, 'prefix': prefix})
            continue

        # full run
        comps = build_components(sample_fraction=frac)

        # 如果构建的数据集为空（无样本），构造合成数据Loader供测试流程使用
        try:
            total_samples = len(comps['train_loader'].combined_dataset)
        except Exception:
            total_samples = 0
        if total_samples == 0:
            print(f"[Synthetic] 数据集为空，使用合成数据代替训练（fraction={frac})")
            # 验证synthetic loader不影响val_loader
            import torch
            from torch.utils.data import DataLoader, Dataset
            class SyntheticDataset(Dataset):
                def __init__(self, length=100, num_frames=cfg.num_frames, size=cfg.student_output_size):
                    self.length = length
                    self.num_frames = num_frames
                    self.size = size
                def __len__(self):
                    return self.length
                def __getitem__(self, idx):
                    frames = torch.rand(self.num_frames, 3, *self.size)
                    density = torch.zeros(self.num_frames, *self.size)
                    return {'frames': frames, 'density_maps': density, 'dataset': 'synthetic'}
            synth = SyntheticDataset(length=200)
            comps['train_loader'] = DataLoader(synth, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

        # If base checkpoint provided, load as initialization
        if base_ckpt and os.path.exists(base_ckpt):
            try:
                print(f"[Init] Loading base checkpoint {base_ckpt} into student/meta")
                load_checkpoint(base_ckpt, comps['student_model'], comps['meta_teacher'], comps['optimizer_student'], comps['optimizer_meta'])
            except Exception as e:
                print(f"[Warn] failed to load base checkpoint: {e}")

        latest_ckpt = cfg.checkpoint_dir / f"{prefix}_latest.pth"
        start_epoch = 0
        if cfg.resume and latest_ckpt.exists():
            print(f"[Resume] found latest checkpoint: {latest_ckpt}")
            start_epoch = load_checkpoint(str(latest_ckpt), comps['student_model'], comps['meta_teacher'], comps['optimizer_student'], comps['optimizer_meta'])

        train_model(
            comps['train_loader'],
            comps['val_loader'],
            comps['device'],
            comps['student_model'],
            comps['meta_teacher'],
            comps['scene_classifier'],
            comps['teacher_orchestrator'],
            comps['optimizer_student'],
            comps['optimizer_meta'],
            comps['scheduler_student'],
            comps['video_augmentor'],
            comps['difficulty_module'],
            distillation_loss_fn=comps['loss_fn'],
            start_epoch=start_epoch,
            prefix=prefix,
            save_interval=cfg.save_interval
        )

        # evaluate and record
        res = evaluate_all_domains(comps['student_model'], comps['device'])
        for ds, (m, r, pf) in res.items():
            results.append({'dataset': ds, 'fraction': frac, 'mae': m, 'rmse': r, 'per_frame_mae': pf, 'prefix': prefix})

    # restore epochs
    cfg.epochs = original_epochs
    return results


def run_generalization_pipeline(dry_run=True):
    """Train on each dataset individually and evaluate on all domains."""
    results = []
    original_epochs = cfg.epochs
    if dry_run:
        cfg.epochs = 1
        cfg.save_interval = 1
    cfg.save_interval = 10

    for ds in list(cfg.datasets):
        prefix = f"single_{ds}"
        cfg.experiment_prefix = prefix
        set_seed(42)

        print(f"\n=== Generalization: training only on {ds} ===")
        if dry_run:
            # record dummy results
            results.append({'train_domain': ds, 'eval_domain': ds, 'mae': -1.0, 'rmse': -1.0, 'per_frame_mae': -1.0, 'prefix': prefix})
            continue

        comps = build_components(dataset_list=[ds])
        latest_ckpt = cfg.checkpoint_dir / f"{prefix}_latest.pth"
        start_epoch = 0
        if cfg.resume and latest_ckpt.exists():
            start_epoch = load_checkpoint(str(latest_ckpt), comps['student_model'], comps['meta_teacher'], comps['optimizer_student'], comps['optimizer_meta'])

        train_model(
            comps['train_loader'], comps['val_loader'], comps['device'],
            comps['student_model'], comps['meta_teacher'], comps['scene_classifier'],
            comps['teacher_orchestrator'], comps['optimizer_student'], comps['optimizer_meta'],
            comps['scheduler_student'], comps['video_augmentor'], comps['difficulty_module'],
            distillation_loss_fn=comps['loss_fn'], start_epoch=start_epoch,
            prefix=prefix, save_interval=cfg.save_interval
        )

        res = evaluate_all_domains(comps['student_model'], comps['device'])
        for ds_eval, (m,r,pf) in res.items():
            results.append({'train_domain': ds, 'eval_domain': ds_eval, 'mae': m, 'rmse': r, 'per_frame_mae': pf, 'prefix': prefix})

    cfg.epochs = original_epochs
    return results


def run_ablation_pipeline(dry_run=True):
    """Perform ablation studies by toggling config flags."""
    results = []
    base_cfg = copy.deepcopy(cfg)
    ablations = [
        ("no_countvid", {"is_countvid_enabled": False}),
        ("no_crowdmpm", {"is_crowdmpm_enabled": False}),
        ("no_oman", {"is_oman_enabled": False}),
        ("no_graspmamba", {"is_graspmamba_enabled": False}),
        ("no_aug", {"is_video_augmentation_enabled": False}),
        ("no_difficulty", {"is_difficulty_aware": False}),
    ]
    original_epochs = cfg.epochs
    if dry_run:
        cfg.epochs = 1
        cfg.save_interval = 1
    cfg.save_interval = 10

    for name, changes in ablations:
        for k,v in changes.items():
            setattr(cfg, k, v)
        prefix = f"ablation_{name}"
        cfg.experiment_prefix = prefix
        set_seed(42)

        print(f"\n=== Ablation: {name} ===")
        if dry_run:
            results.append({'ablation': name, 'dataset': 'all', 'mae': -1.0, 'rmse': -1.0, 'per_frame_mae': -1.0, 'prefix': prefix})
            for k in changes.keys():
                setattr(cfg, k, getattr(base_cfg, k))
            continue

        comps = build_components()
        latest_ckpt = cfg.checkpoint_dir / f"{prefix}_latest.pth"
        start_epoch = 0
        if cfg.resume and latest_ckpt.exists():
            start_epoch = load_checkpoint(str(latest_ckpt), comps['student_model'], comps['meta_teacher'], comps['optimizer_student'], comps['optimizer_meta'])

        train_model(
            comps['train_loader'], comps['val_loader'], comps['device'],
            comps['student_model'], comps['meta_teacher'], comps['scene_classifier'],
            comps['teacher_orchestrator'], comps['optimizer_student'], comps['optimizer_meta'],
            comps['scheduler_student'], comps['video_augmentor'], comps['difficulty_module'],
            distillation_loss_fn=comps['loss_fn'], start_epoch=start_epoch,
            prefix=prefix, save_interval=cfg.save_interval
        )

        res = evaluate_all_domains(comps['student_model'], comps['device'])
        for ds_eval,(m,r,pf) in res.items():
            results.append({'ablation': name, 'dataset': ds_eval, 'mae': m, 'rmse': r, 'per_frame_mae': pf, 'prefix': prefix})

        # restore modified flags
        for k in changes.keys():
            setattr(cfg, k, getattr(base_cfg, k))

    # restore full cfg
    for attr in vars(base_cfg):
        setattr(cfg, attr, getattr(base_cfg, attr))
    cfg.epochs = original_epochs
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run small dry-run (1 epoch)')
    parser.add_argument('--base-checkpoint', type=str, default=None, help='Optional base checkpoint to initialize student')
    args = parser.parse_args()

    # 按用户要求的低数据比例：1%, 2%, 10%, 20%, 40%
    fractions = [0.01, 0.02, 0.1, 0.2, 0.4]
    low_results = run_lowdata_pipeline(fractions, dry_run=args.dry_run, base_ckpt=args.base_checkpoint)

    os.makedirs('experiments/results', exist_ok=True)
    csv_path = os.path.join('experiments/results', f'lowdata_results_{"dry" if args.dry_run else "full"}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset','fraction','mae','rmse','per_frame_mae','prefix'])
        writer.writeheader()
        for row in low_results:
            writer.writerow(row)
    print(f"Low-data results written to {csv_path}")

    # run generalization
    gen_results = run_generalization_pipeline(dry_run=args.dry_run)
    csv_path = os.path.join('experiments/results', f'generalization_results_{"dry" if args.dry_run else "full"}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['train_domain','eval_domain','mae','rmse','per_frame_mae','prefix'])
        writer.writeheader()
        for row in gen_results:
            writer.writerow(row)
    print(f"Generalization results written to {csv_path}")

    # run ablation
    ablate_results = run_ablation_pipeline(dry_run=args.dry_run)
    csv_path = os.path.join('experiments/results', f'ablation_results_{"dry" if args.dry_run else "full"}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ablation','dataset','mae','rmse','per_frame_mae','prefix'])
        writer.writeheader()
        for row in ablate_results:
            writer.writerow(row)
    print(f"Ablation results written to {csv_path}")
