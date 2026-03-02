"""
蒸馏损失模块
修复：尺寸对齐、逐样本temporal、反向蒸馏→轻量校准
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class DistillationLoss(nn.Module):
    """综合蒸馏损失"""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_output, teacher_output, lambda_kl, epoch,
                student_confidence=None):
        """
        Args:
            student_output: Dict {'density_map': (B,T,H,W), ...}
            teacher_output: Dict {'density_map': (B,T,H,W), ...}
            lambda_kl: (B, 1) 或 scalar
            epoch: int
            student_confidence: (B,)
        Returns:
            total_loss, loss_dict
        """
        losses = {}

        if teacher_output is None:
            # 无教师，仅自监督（暂时返回零损失）
            return torch.tensor(0.0, device=student_output['density_map'].device), {}

        # 1. 尺寸对齐
        student_density = student_output['density_map']  # (B, T, H, W)
        teacher_density = teacher_output['density_map']  # (B, T, H, W)

        if student_density.shape != teacher_density.shape:
            teacher_density = F.interpolate(
                teacher_density.view(-1, 1, *teacher_density.shape[2:]),
                size=student_density.shape[2:],
                mode='bilinear',
                align_corners=False
            ).view(student_density.shape[0], student_density.shape[1], *student_density.shape[2:])

        # 2. L1 损失
        l1_loss = self.l1_loss(student_density, teacher_density)
        losses['l1'] = cfg.lambda_l1 * l1_loss

        # 3. KL 散度损失
        student_prob = F.softmax(student_density.flatten(2), dim=2)
        teacher_prob = F.softmax(teacher_density.flatten(2), dim=2)

        kl_loss = self.kl_loss(
            F.log_softmax(student_density.flatten(2), dim=2),
            teacher_prob
        )

        if isinstance(lambda_kl, torch.Tensor):
            lambda_kl_mean = lambda_kl.mean()
        else:
            lambda_kl_mean = lambda_kl

        losses['kl'] = lambda_kl_mean * kl_loss

        # 4. 时序一致性损失（逐样本计算）
        temporal_loss = self.temporal_consistency_loss(student_density, teacher_density)
        losses['temporal'] = cfg.lambda_temp_consistency * temporal_loss

        # 5. Flux 保守损失（仅当教师输出包含 flux 时使用）
        if 'flux' in teacher_output:
            teacher_flux = teacher_output['flux']
            student_flux = student_output.get('flux', torch.zeros_like(teacher_flux))
            try:
                flux_loss = self.l1_loss(student_flux, teacher_flux)
                losses['flux'] = cfg.lambda_flux_conservation * flux_loss
            except Exception:
                # shape mismatch or missing student flux,跳过
                pass

        # 6. 轻量校准（改名，不再叫"反向蒸馏"）
        if epoch >= cfg.start_calibration_epoch:
            lambda_calib = self.compute_calibration_weight(epoch, student_confidence)
            if lambda_calib > 0:
                calib_loss = self.calibration_loss(teacher_density, student_density)
                losses['calibration'] = lambda_calib * calib_loss

        total_loss = sum(losses.values())
        return total_loss, losses

    def temporal_consistency_loss(self, student_density, teacher_density):
        """
        时序一致性（逐样本计算再平均）

        Args:
            student_density: (B, T, H, W)
            teacher_density: (B, T, H, W)
        """
        B, T = student_density.shape[:2]

        if T < 2:
            return torch.tensor(0.0, device=student_density.device)

        student_diffs = []
        teacher_diffs = []

        for b in range(B):
            s_diff = torch.abs(student_density[b, 1:] - student_density[b, :-1]).mean()
            t_diff = torch.abs(teacher_density[b, 1:] - teacher_density[b, :-1]).mean()
            student_diffs.append(s_diff)
            teacher_diffs.append(t_diff)

        student_smoothness = torch.stack(student_diffs).mean()
        teacher_smoothness = torch.stack(teacher_diffs).mean()

        return F.mse_loss(student_smoothness, teacher_smoothness)

    def compute_calibration_weight(self, epoch, student_confidence=None):
        """计算校准权重（基于学生置信度）"""
        progress = (epoch - cfg.start_calibration_epoch) / (cfg.epochs - cfg.start_calibration_epoch + 1)
        lambda_base = cfg.lambda_calibration_max * min(progress, 1.0)

        if student_confidence is not None:
            high_conf_mask = (student_confidence > cfg.calibration_confidence_threshold).float()
            adjustment = high_conf_mask.mean()
            lambda_calib = lambda_base * (1 + adjustment)
        else:
            lambda_calib = lambda_base

        return float(lambda_calib)

    def calibration_loss(self, teacher_density, student_density):
        """
        校准损失（学生→教师，但实际只用于调整门控/温度等轻量参数）
        这里仅计算一致性，梯度限制在外部实现
        """
        return F.mse_loss(teacher_density, student_density)


def compute_student_confidence(student_output):
    """学生置信度（密度图方差的倒数）"""
    density = student_output['density_map']
    variance = density.var(dim=[1, 2, 3])
    confidence = torch.exp(-variance)
    return confidence