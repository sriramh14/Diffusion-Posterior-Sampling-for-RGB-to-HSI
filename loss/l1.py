"""L1 reconstruction loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    return F.l1_loss(pred, target, reduction=reduction)


class L1ReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return l1_loss(pred, target, reduction=self.reduction)
