"""Root-mean-squared error metric."""

from __future__ import annotations

import torch


def rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    mse_value = torch.mean((pred - target).square())
    if eps > 0:
        mse_value = torch.clamp(mse_value, min=eps)
    return torch.sqrt(mse_value)
