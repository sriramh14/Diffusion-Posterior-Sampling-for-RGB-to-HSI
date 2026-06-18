"""Shared preprocessing for reconstruction metrics."""

from __future__ import annotations

from typing import Tuple

import torch


def prepare_metric_tensors(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Clamp predictions only when targets are already in the [0, 1] range."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )

    target_min = float(target.detach().amin().item())
    target_max = float(target.detach().amax().item())

    if target_min >= -1e-6 and target_max <= 1.0 + 1e-6:
        return pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0), 1.0

    data_range = max(target_max - target_min, eps)
    return pred, target, data_range
