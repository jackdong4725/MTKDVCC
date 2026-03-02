"""单域训练及泛化评估脚本

将模型分别仅在每个数据集上训练，然后在所有数据集的验证集上评估性能。
"""
import sys
import os
import torch

# Ensure project root is on sys.path so top-level imports (e.g. `from config import cfg`) work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg
from experiments.utils import build_components, set_seed, evaluate_all_domains
from main import train_model, load_checkpoint


def run_generalization_experiments():
    cfg.save_interval = 10
    # 同一个前缀下可能中断后继续训练
    cfg.resume = True

    for ds in list(cfg.datasets):
        prefix = f"single_{ds}"
        cfg.experiment_prefix = prefix
        set_seed(42)

        print(f"\n=== 在单域 {ds} 上训练 ({prefix}) ===")
        comps = build_components(dataset_list=[ds])

        # 如果提供了 base checkpoint（通过环境变量或外部调用传入 cfg.base_checkpoint），先尝试加载
        if getattr(cfg, 'base_checkpoint', None):
            base_ckpt = cfg.base_checkpoint
            if os.path.exists(base_ckpt):
                try:
                    print(f"[Init] 使用 base checkpoint {base_ckpt} 初始化模型和 meta")
                    load_checkpoint(base_ckpt, comps['student_model'], comps['meta_teacher'], comps['optimizer_student'], comps['optimizer_meta'])
                except Exception as e:
                    print(f"[Warn] 加载 base checkpoint 失败: {e}")

        # 尝试恢复已有训练进度
        latest_ckpt = cfg.checkpoint_dir / f"{prefix}_latest.pth"
        start_epoch = 0
        if cfg.resume and latest_ckpt.exists():
            print(f"[恢复] 发现已有 checkpoint: {latest_ckpt}")
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

        # 在所有域上评估
        results = evaluate_all_domains(comps['student_model'], comps['device'])
        print(f"训练于 {ds} 后的跨域评估结果:")
        for ds_eval, (m, r, pf) in results.items():
            print(f"  {ds_eval}: MAE={m:.2f}, RMSE={r:.2f}, per-frame MAE={pf:.2f}")


if __name__ == "__main__":
    run_generalization_experiments()
