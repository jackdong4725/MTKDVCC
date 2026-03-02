"""Smoke test: run 1 epoch low-data (1%) training to validate pipeline."""
import os
import sys
import traceback

# ensure project root is importable so local modules (config, utils, models) can be found
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import cfg
from experiments.utils import build_components, evaluate_all_domains
from main import train_model, save_checkpoint

# robustly import set_seed from utils/helpers.py to avoid conflicts with other utils modules
import importlib.util
from pathlib import Path
helpers_path = Path(project_root) / 'utils' / 'helpers.py'
if helpers_path.exists():
    spec = importlib.util.spec_from_file_location('project_utils_helpers', str(helpers_path))
    _helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_helpers)
    set_seed = _helpers.set_seed
else:
    from utils.helpers import set_seed

set_seed(42)

# light settings for smoke test
cfg.epochs = 1
cfg.eval_interval = 100
cfg.save_interval = 100
cfg.use_wandb = False
cfg.batch_size = 1
cfg.resume = False

print("[smoke] 构建组件 (sample_fraction=0.01)")
components = build_components(sample_fraction=0.01)

device = components['device']
train_loader = components['train_loader']
val_loader = components['val_loader']
scene_classifier = components['scene_classifier']
meta_teacher = components['meta_teacher']
teacher_orchestrator = components['teacher_orchestrator']
student_model = components['student_model']
optimizer_student = components['optimizer_student']
optimizer_meta = components['optimizer_meta']
scheduler_student = components['scheduler_student']
video_augmentor = components['video_augmentor']
difficulty_module = components['difficulty_module']
loss_fn = components['loss_fn']

print(f"[smoke] device={device}, train_batches={len(train_loader)}")

try:
    trained = train_model(
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
        distillation_loss_fn=loss_fn,
        start_epoch=0,
        prefix='smoke_lowdata_1pct',
        save_interval=cfg.save_interval
    )

    print('[smoke] 训练完成，开始评估 (若有验证集)')
    try:
        results = evaluate_all_domains(trained, device)
        print('[smoke] 评估结果:', results)
    except Exception as e:
        print('[smoke] 评估失败:', e)

    # 保存最终模型
    try:
        save_checkpoint(0, trained, meta_teacher, optimizer_student, optimizer_meta, prefix='smoke_lowdata_1pct', filename='smoke_final.pth')
    except Exception as e:
        print('[smoke] 保存检查点失败:', e)

except Exception as e:
    print('[smoke] 训练中发生异常:')
    traceback.print_exc()
    raise

print('[smoke] 完成')
