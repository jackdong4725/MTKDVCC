"""
MT-FKD 全局配置文件
学生模型更新为 PointDGMamba
专家权重文件路径已根据您的最新信息更新。
"""
import torch
from pathlib import Path


class Config:
    # ==================== 项目路径 ====================
    project_root = Path("/root/autodl-tmp/MT-FKD")
    weights_dir = project_root / "weights"

    # ==================== 数据集配置 ====================
    # 仅使用论文中提到的三个数据集进行演化增强和训练
    datasets = ['FDST', 'MALL', 'Venice']
    data_root = Path("/root/autodl-tmp/MT-FKD/datasets/")
    img_size = (768, 768)
    num_frames = 8

    # ==================== SOTA 专家集群开关 ====================
    is_countvid_enabled = True
    is_langmamba_enabled = False
    # RefAtomNet (refAVA2) 已从项目中移除
    is_refatomnet_enabled = False
    is_crowdmpm_enabled = True
    is_oman_enabled = True
    is_pointdgmamba_student_enabled = True

    # ==================== 学生模型: PointDGMamba 配置 ====================
    pointdgmamba_root = project_root / "PointDGMamba"
    student_checkpoint = weights_dir / "pointdgmamba_student.pth" # 学生模型权重 (由本项目训练生成)

    # CNN 特征提取器配置
    student_cnn_out_dim = 384

    # PointNet++ 采样配置
    student_num_points = 2048

    # PointDGMamba 核心配置
    student_embed_dim = 384
    student_depth = 12
    student_d_state = 16
    student_d_conv = 4
    student_expand = 2
    student_drop_path_rate = 0.1

    # Domain Generalization 配置
    student_num_domains = 7
    student_use_domain_token = True
    student_domain_embed_dim = 64 # Not used directly in this version, but keep for consistency
    student_domain_reduction = 4

    # 密度图解码器配置
    student_decoder_dim = 256
    student_output_size = (192, 192)

    # ==================== CountVid 配置 ====================
    countvid_root = project_root / "CountVid"
    # ✅ CountVid 依赖 GroundingDINO (countgd_box.pth) 和 SAM (sam2.1_hiera_large.pt)
    # GroundingDINO 配置文件（项目内已有配置位于 CountVid/tools）
    groundingdino_config_file = countvid_root / "tools/GroundingDINO_SwinB_cfg.py"
    groundingdino_checkpoint = weights_dir / "groundingdino_swint_ogc.pth"  # 使用论文指定的权重文件名

    # GroundingDINO 使用的 BERT 模型应放在此目录下，以便离线加载
    groundingdino_bert_path = project_root / "bert-base-uncased"

    countvid_box_threshold = 0.35
    countvid_text_threshold = 0.25
    countvid_default_prompt = "person"
    countvid_nms_threshold = 0.8

    sam_checkpoint = weights_dir / "sam2.1_hiera_large.pt" # ✅ 更新
    sam_model_type = "vit_h" # Still assume ViT-H for SAM

    # ==================== LangMamba 配置 ====================
    langmamba_root = project_root / "LangMamba"
    langmamba_checkpoint = weights_dir / "vgg.pth" # ✅ 更新 (注意：vgg.pth 似乎不是 LangMamba 官方权重，后续可能需要手动调整)

    # Vision Mamba 配置
    langmamba_vision_encoder = "vim_small"
    langmamba_img_size = 224
    langmamba_patch_size = 16
    langmamba_embed_dim = 384
    langmamba_depth = 24
    langmamba_num_heads = 6
    langmamba_mlp_ratio = 4.0

    # Language Model 配置
    langmamba_text_encoder = "bert-base-uncased"
    langmamba_max_text_len = 77

    # Fusion 配置
    langmamba_fusion_layers = 4
    langmamba_num_queries = 100

    # RefAtomNet 已从项目中移除，相关配置不再使用

    # ==================== OMAN 配置 ====================
    oman_root = project_root / "OMAN"
    oman_checkpoint = weights_dir / "convnext_small_384_in22ft1k.pth" # ✅ 更新 (这看起来是 backbone 权重)

    oman_hidden_dim = 256
    oman_num_encoder_layers = 6
    oman_nheads = 8
    oman_dim_feedforward = 2048
    oman_dropout = 0.1

    oman_num_ref_points = 50
    oman_num_indep_points = 20
    oman_window_size = [32, 32, 32, 32]

    # ==================== GraspMamba 配置 ====================
    is_graspmamba_enabled = True
    graspmamba_root = project_root / "GraspMamba"
    graspmamba_checkpoint = weights_dir / "graspmamba.pth"  # 占位权重路径（请手动替换为真实权重）
    graspmamba_input_size = 224
    graspmamba_confidence_threshold = 0.2

    # ==================== CrowdMPM 配置 ====================
    crowdmpm_root = project_root / "CrowdMPM"
    # ✅ CrowdMPM 权重由 mpm_best.pth 和 multi_cvae_2000.pth 组成
    crowdmpm_checkpoint = weights_dir / "mpm_best.pth" # ✅ 更新 (主模型权重)
    crowdmpm_alpha_checkpoint = weights_dir / "mpm_best.pth" # ✅ 假设 Alpha ParaNet 也在主模型中
    crowdmpm_E_K_checkpoint = weights_dir / "mpm_best.pth" # ✅ 假设 E_K ParaNet 也在主模型中
    crowdmpm_cvae_checkpoint = weights_dir / "multi_cvae_2000.pth" # ✅ 更新 (CVAE 权重)

    # MPM Simulation 配置
    crowdmpm_n_substeps = 60
    crowdmpm_n_particles_sample = 500
    crowdmpm_flow_threshold = 1.0

    # Grid 配置
    mpm_n_grid = (30, 20)
    mpm_res = (3.0, 2.0)
    mpm_dx = mpm_res[0] / mpm_n_grid[0]
    mpm_inv_dx = 1.0 / mpm_dx
    mpm_dt = 1.0 / 60.0

    # Particle 配置
    mpm_p_vol = (mpm_dx * 0.5) ** 2
    mpm_p_mass = mpm_p_vol * 1.0
    mpm_p_radius = 0.025

    # Boundary 配置
    mpm_bound = 3
    mpm_goal = [1.5, 2.2]
    mpm_door_l_pos = 1.2
    mpm_door_r_pos = 1.8

    # Physical 配置
    mpm_E = 2000.0
    mpm_K = 3.0
    mpm_w_ext = 0.03

    # Social Force 配置
    theta_1 = 0.05
    theta_2 = 1.5
    comfort_dis = 0.3

    # CVAE 配置
    mpm_n_decoder_cvae = 4

    # ==================== 场景感知分类器 ====================
    num_scene_classes = 7
    meta_input_dim = 135
    scene_classifier_hidden = [256, 128]

    SCENE_LABELS = [
        "Extreme_Dense_Static",
        "High_Dense_Directional",
        "Medium_Dense_Chaotic",
        "Low_Dense_Sparse",
        "Abrupt_Change",
        "Open_World_Objects",
        "Uncertain_Mixed"
    ]

    # ==================== 元教师生成器 ====================
    is_meta_enabled = True
    meta_output_dim = 128
    meta_hidden_dims = [512, 256]
    lr_meta = 5e-5

    # ==================== 教师选择策略 ====================
    teacher_strategy = "soft"
    teacher_selection_temperature = 1.0
    teacher_top_k = 2

    # ==================== 蒸馏损失配置 ====================
    lambda_kl_initial = 0.5
    lambda_temp_consistency = 0.2
    lambda_l1 = 1.0
    lambda_flux_conservation = 0.1

    # 可选基础 checkpoint（用于 generalization/ablation 从 low-data 权重初始化）
    base_checkpoint = None
    start_calibration_epoch = 100
    lambda_calibration_max = 0.3
    calibration_confidence_threshold = 0.7

    # ==================== 难度感知模块 ====================
    is_difficulty_aware = True
    difficulty_update_freq = 5
    difficulty_ema_alpha = 0.9
    difficulty_lambda_scale = 2.0

    # ==================== 数据增强 (演化式增强 VideoMix/CutMix) ====================
    is_video_augmentation_enabled = True
    videomix_prob = 0.5  # 触发演化式增强的概率
    videomix_alpha = 1.0  # Beta 分布参数，用于混合权重

    # ==================== 训练配置 ====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    batch_size = 4
    lr = 1e-4
    weight_decay = 1e-5
    epochs = 300
    warmup_epochs = 10

    optimizer_type = "AdamW"
    scheduler_type = "CosineAnnealingWarmRestarts"

    save_interval = 50
    eval_interval = 20
    checkpoint_dir = project_root / "checkpoints"

    # 实验控制辅助字段
    resume = False  # 是否从上次 checkpoint 恢复
    experiment_prefix = "run"  # 用于区分不同实验的前缀

    # ==================== 教师离线缓存 ====================
    use_teacher_cache = False
    teacher_cache_dir = project_root / "teacher_cache"

    # ==================== 可视化与导出 ====================
    results_dir = project_root / "results"
    save_mat_data = True
    visualization_dpi = 600

    # ==================== 理论验证实验配置 ====================
    info_gain_eval_samples = 1000
    track_hypothesis_complexity = True

    # ==================== 日志配置 ====================
    use_wandb = True
    wandb_project = "MT-FKD"
    wandb_entity = "your_entity"
    log_interval = 10

    def __init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.use_teacher_cache:
            self.teacher_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_enabled_experts(self):
        """返回已启用的专家列表"""
        experts = []
        if self.is_countvid_enabled:
            experts.append("CountVid")
        if self.is_langmamba_enabled:
            experts.append("LangMamba")
        # RefAtomNet 已从项目中移除
        if self.is_crowdmpm_enabled:
            experts.append("CrowdMPM")
        if self.is_oman_enabled:
            experts.append("OMAN")
        return experts

    def validate_config(self):
        """配置验证"""
        assert self.is_pointdgmamba_student_enabled, "学生模型必须启用"

        # 检查 CountVid 依赖
        if self.is_countvid_enabled:
            if not self.groundingdino_config_file.exists():
                print(f"[警告] GroundingDINO 配置不存在: {self.groundingdino_config_file}")
            # ✅ 增加对 GroundingDINO 和 SAM 权重文件的存在性检查
            if not self.groundingdino_checkpoint.exists():
                print(f"[警告] CountVid 的 GroundingDINO 权重不存在: {self.groundingdino_checkpoint}")
            if not self.sam_checkpoint.exists():
                print(f"[警告] CountVid 的 SAM 权重不存在: {self.sam_checkpoint}")

        # 检查 LangMamba 依赖
        if self.is_langmamba_enabled:
            if not self.langmamba_root.exists():
                print(f"[警告] LangMamba 目录不存在: {self.langmamba_root}")
            # ✅ 增加对 LangMamba 权重文件的存在性检查
            if not self.langmamba_checkpoint.exists():
                print(f"[警告] LangMamba 权重不存在: {self.langmamba_checkpoint}")

        # 检查 OMAN 依赖
        if self.is_oman_enabled:
            if not self.oman_root.exists():
                print(f"[警告] OMAN 目录不存在: {self.oman_root}")
            # ✅ 增加对 OMAN 权重文件的存在性检查
            if not self.oman_checkpoint.exists():
                print(f"[警告] OMAN 权重不存在: {self.oman_checkpoint}")

        # 检查 GraspMamba 依赖
        if self.is_graspmamba_enabled:
            if not self.graspmamba_root.exists():
                print(f"[警告] GraspMamba 目录不存在: {self.graspmamba_root}")
            if not self.graspmamba_checkpoint.exists():
                print(f"[警告] GraspMamba 权重不存在 (占位): {self.graspmamba_checkpoint}")

        # 检查 CrowdMPM 依赖
        if self.is_crowdmpm_enabled:
            if not self.crowdmpm_root.exists():
                print(f"[警告] CrowdMPM 目录不存在: {self.crowdmpm_root}")
            # ✅ 增加对 CrowdMPM 权重文件的存在性检查
            if not self.crowdmpm_checkpoint.exists():
                print(f"[警告] CrowdMPM 主权重 (mpm_best.pth) 不存在: {self.crowdmpm_checkpoint}")
            if not self.crowdmpm_cvae_checkpoint.exists():
                print(f"[警告] CrowdMPM CVAE 权重 (multi_cvae_2000.pth) 不存在: {self.crowdmpm_cvae_checkpoint}")


        if len(self.get_enabled_experts()) == 0:
            print("[警告] 无可用专家，将使用学生自学习模式")

        print(f"[Config] 已启用专家: {self.get_enabled_experts()}")
        print(f"[Config] 学生模型: PointDGMamba")
        print(f"[Config] 设备: {self.device} ({self.num_gpus} GPUs)")

    def to_serializable_dict(self):
        """转换为可序列化的字典"""
        return {
            'datasets': self.datasets,
            'img_size': self.img_size,
            'num_frames': self.num_frames,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'epochs': self.epochs,
            'teacher_strategy': self.teacher_strategy,
            'student_output_size': self.student_output_size,
            'enabled_experts': self.get_enabled_experts(),
            'is_meta_enabled': self.is_meta_enabled,
            'is_difficulty_aware': self.is_difficulty_aware,
            'student_model': 'PointDGMamba'
        }


# 全局配置实例
cfg = Config()
cfg.validate_config()

