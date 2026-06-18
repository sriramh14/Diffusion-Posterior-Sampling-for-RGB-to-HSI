"""Band-wise local SSIM averaged over a hyperspectral batch."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 3,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute local SSIM over ``[B, C, H, W]`` tensors.

    Each spectral band is treated as an image channel and the final value is
    averaged over all batches, bands, and spatial positions.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    if pred.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], received {pred.shape}")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)

    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    sigma_pred = (
        F.avg_pool2d(pred.square(), window_size, stride=1, padding=padding)
        - mu_pred_sq
    )
    sigma_target = (
        F.avg_pool2d(target.square(), window_size, stride=1, padding=padding)
        - mu_target_sq
    )
    sigma_cross = (
        F.avg_pool2d(pred * target, window_size, stride=1, padding=padding)
        - mu_cross
    )

    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (
        (mu_pred_sq + mu_target_sq + c1)
        * (sigma_pred + sigma_target + c2)
    )
    return (numerator / torch.clamp(denominator, min=eps)).mean()
