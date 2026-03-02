"""
专家模型模块
包含所有真实实现的专家（无占位符）
"""
from config import cfg

# CountVid
from .countvid import CountVid

# OMAN
if cfg.is_oman_enabled:
    try:
        from .oman import OMAN, build_oman_expert
    except ImportError as e:
        print(f"[警告] OMAN 导入失败: {e}")
        OMAN = None

# CrowdMPM
if cfg.is_crowdmpm_enabled:
    try:
        from .crowdmpm import CrowdMPM, build_crowdmpm_expert
    except ImportError as e:
        print(f"[警告] CrowdMPM 导入失败: {e}")
        CrowdMPM = None

# RefAtomNet 已从项目中移除，不进行导入

# LangMamba 已替换为 GraspMamba（zero-shot），不再直接导入 LangMamba

# GraspMamba
if cfg.is_graspmamba_enabled:
    try:
        from .graspmamba import GraspMambaExpert, build_graspmamba_expert
    except ImportError as e:
        print(f"[警告] GraspMamba 导入失败: {e}")
        GraspMambaExpert = None


__all__ = ['CountVid']

if cfg.is_oman_enabled and OMAN is not None:
    __all__.extend(['OMAN', 'build_oman_expert'])

if cfg.is_crowdmpm_enabled and CrowdMPM is not None:
    __all__.extend(['CrowdMPM', 'build_crowdmpm_expert'])



if cfg.is_graspmamba_enabled and GraspMambaExpert is not None:
    __all__.extend(['GraspMambaExpert', 'build_graspmamba_expert'])