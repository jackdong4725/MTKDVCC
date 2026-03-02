"""
KL Divergence Loss for Distance Distribution
"""
import torch
import torch.nn.functional as F


def kl_loss_distance(pred_dist, labels):
    """
    计算预测距离分布与GT距离分布之间的KL散度

    Args:
        pred_dist: (N, M) - 预测的距离矩阵
        labels: Dict containing 'gt_fuse_pts0', 'gt_fuse_pts1', 'gt_fuse_num'

    Returns:
        kl_div: scalar
    """
    gt_fuse_pts0 = labels['gt_fuse_pts0'][0][:labels['gt_fuse_num']]
    gt_fuse_pts1 = labels['gt_fuse_pts1'][0][:labels['gt_fuse_num']]

    # 计算GT距离
    gt_dist_list = []
    for i in range(labels['gt_fuse_num']):
        dist = torch.cdist(gt_fuse_pts0[i].unsqueeze(0), gt_fuse_pts1[i].unsqueeze(0))
        gt_dist_list.append(dist)

    if len(gt_dist_list) == 0:
        return 0.0

    gt_dist = torch.cat(gt_dist_list, dim=0)

    # 构建直方图
    num_bins = 10
    min_val = min(pred_dist.min(), gt_dist.min())
    max_val = max(pred_dist.max(), gt_dist.max())

    pred_hist = torch.histc(pred_dist, bins=num_bins, min=min_val, max=max_val)
    pred_dist_prob = pred_hist / (pred_hist.sum() + 1e-8)

    gt_hist = torch.histc(gt_dist, bins=num_bins, min=min_val, max=max_val)
    gt_dist_prob = gt_hist / (gt_hist.sum() + 1e-8)

    # KL散度
    epsilon = 1e-8
    kl_div = F.kl_div(
        (pred_dist_prob + epsilon).log(),
        gt_dist_prob.cuda() + epsilon,
        reduction='sum'
    )

    return kl_div.item()
