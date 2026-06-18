"""Mean Relative Absolute Error for hyperspectral reconstruction."""

from __future__ import annotations

import torch
from torch import nn


def mrae(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute mean relative absolute error.

    Args:
        pred: Reconstructed HSI tensor, normally ``[B, C, H, W]``.
        target: Ground-truth HSI tensor with the same shape as ``pred``.
        eps: Minimum denominator used near zero-valued target elements.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    error = torch.abs(pred - target)
    denominator = torch.clamp(torch.abs(target), min=eps)
    value = error / denominator

    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    if reduction == "none":
        return value
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


class MRAELoss(nn.Module):
    """``nn.Module`` wrapper around :func:`mrae`."""

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mrae(pred, target, eps=self.eps, reduction=self.reduction)
