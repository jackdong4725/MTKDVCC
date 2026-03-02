"""消融实验脚本

依次关闭各个专家以及不同模块，观察性能变化。
"""
import copy
import sys
import os

# Ensure project root is on sys.path so top-level imports (e.g. `from config import cfg`) work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import cfg
from experiments.utils import build_components, set_seed, evaluate_all_domains
from main import train_model, load_checkpoint


def run_ablation_experiments():
    # 保存原始配置以便恢复
    base_cfg = copy.deepcopy(cfg)

    # 要执行的消融组合
    ablations = [
        ("no_countvid", {"is_countvid_enabled": False}),
        ("no_crowdmpm", {"is_crowdmpm_enabled": False}),
        ("no_oman", {"is_oman_enabled": False}),
        ("no_graspmamba", {"is_graspmamba_enabled": False}),
        ("no_aug", {"is_video_augmentation_enabled": False}),
        ("no_difficulty", {"is_difficulty_aware": False}),
    ]

    cfg.save_interval = 10
    cfg.resume = True

    for name, changes in ablations:
        # 应用局部更改
        for k, v in changes.items():
            setattr(cfg, k, v)

        prefix = f"ablation_{name}"
        cfg.experiment_prefix = prefix
        set_seed(42)

        print(f"\n=== 消融设置 {name} ===")
        comps = build_components()

        # 如果提供 base checkpoint（例如来自 low-data 训练），优先加载用于初始化
        start_epoch = 0
        if getattr(cfg, 'base_checkpoint', None) and os.path.exists(cfg.base_checkpoint):
            try:
                print(f"[Init] 使用 base checkpoint {cfg.base_checkpoint} 初始化模型和 meta")
                load_checkpoint(cfg.base_checkpoint, comps['student_model'], comps['meta_teacher'], comps['optimizer_student'], comps['optimizer_meta'])
            except Exception as e:
                print(f"[Warn] 加载 base checkpoint 失败: {e}")

        latest_ckpt = cfg.checkpoint_dir / f"{prefix}_latest.pth"
        if cfg.resume and latest_ckpt.exists():
            print(f"[恢复] 检测到 checkpoint {latest_ckpt}")
            start_epoch = load_checkpoint(
                str(latest_ckpt),
                comps['student_model'],
                comps['meta_teacher'],
                comps['optimizer_student'],
                comps['optimizer_meta']
            )

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

        results = evaluate_all_domains(comps['student_model'], comps['device'])
        print(f"[{prefix}] 消融后跨域评估:")
        for ds, (m, r, pf) in results.items():
            print(f"  {ds}: MAE={m:.2f}, RMSE={r:.2f}, per-frame MAE={pf:.2f}")

        # 恢复原始配置
        for k in changes.keys():
            setattr(cfg, k, getattr(base_cfg, k))

    # 恢复 cfg 的全部属性
    for attr in vars(base_cfg):
        setattr(cfg, attr, getattr(base_cfg, attr))


if __name__ == "__main__":
    run_ablation_experiments()
