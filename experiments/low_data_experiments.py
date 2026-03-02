"""低数据比例训练实验脚本

每个设置都会在不同前缀下保存 checkpoint，可通过配置 resume=True 继续训练。
"""
import os
import sys
import torch

# Ensure project root is on sys.path so top-level imports (e.g. `from config import cfg`) work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg
from experiments.utils import build_components, set_seed, evaluate_all_domains
from main import train_model, load_checkpoint


def run_low_data_experiments():
    fractions = [0.01, 0.05, 0.1, 0.2, 0.4]
    # 为了频繁保存，我们将间隔调小
    cfg.save_interval = 10

    for frac in fractions:
        prefix = f"lowdata_{int(frac*100)}pct"
        cfg.experiment_prefix = prefix
        # 如果已经存在 checkpoint 则尝试恢复
        cfg.resume = True
        set_seed(42)

        print(f"\n=== 训练 {frac*100:.0f}% 数据 ({prefix}) ===")
        comps = build_components(sample_fraction=frac)

        # 检查是否有可恢复的最新 checkpoint
        latest_ckpt = cfg.checkpoint_dir / f"{prefix}_latest.pth"
        start_epoch = 0
        if cfg.resume and latest_ckpt.exists():
            print(f"[恢复] 检测到已有 checkpoint: {latest_ckpt}")
            start_epoch = load_checkpoint(str(latest_ckpt),
                                          comps['student_model'],
                                          comps['meta_teacher'],
                                          comps['optimizer_student'],
                                          comps['optimizer_meta'])

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

        # 评估在所有域上的表现（一般会与训练域一致）
        results = evaluate_all_domains(comps['student_model'], comps['device'])
        print(f"[{prefix}] 验证集评估结果:")
        for ds, (m, r, pf) in results.items():
            print(f"  {ds}: MAE={m:.2f}, RMSE={r:.2f}, per-frame MAE={pf:.2f}")


if __name__ == "__main__":
    run_low_data_experiments()
