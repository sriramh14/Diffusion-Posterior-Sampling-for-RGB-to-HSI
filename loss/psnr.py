"""Peak signal-to-noise ratio metric."""

from __future__ import annotations

import torch


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    if data_range <= 0:
        raise ValueError("data_range must be positive")

    mse_value = torch.mean((pred - target).square())
    max_value = torch.as_tensor(
        data_range * data_range,
        dtype=pred.dtype,
        device=pred.device,
    )
    return 10.0 * torch.log10(max_value / torch.clamp(mse_value, min=eps))
