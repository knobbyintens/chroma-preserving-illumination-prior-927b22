"""Losses for illumination consistency and chroma preservation."""

import torch
import torch.nn.functional as F


def illumination_consistency(illumination: torch.Tensor, target: torch.Tensor, eps: float = 1e-3):
    gt_illum = target.mean(dim=1, keepdim=True).clamp(min=eps)
    return F.l1_loss(torch.log(illumination + eps), torch.log(gt_illum + eps))


def chroma_preservation(pred: torch.Tensor, target: torch.Tensor):
    """Penalise hue drift by comparing normalised RGB chroma components."""
    pred_n = pred / pred.sum(dim=1, keepdim=True).clamp(min=1e-3)
    tgt_n = target / target.sum(dim=1, keepdim=True).clamp(min=1e-3)
    return F.l1_loss(pred_n, tgt_n)


def total_loss(pred, target, illumination, weights=None):
    weights = weights or {"pixel": 1.0, "illum": 0.2, "chroma": 0.35}
    return (
        weights["pixel"] * F.l1_loss(pred, target)
        + weights["illum"] * illumination_consistency(illumination, target)
        + weights["chroma"] * chroma_preservation(pred, target)
    )
