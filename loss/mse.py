"""Mean-squared reconstruction loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    return F.mse_loss(pred, target, reduction=reduction)


class MSEReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mse_loss(pred, target, reduction=self.reduction)
