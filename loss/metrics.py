"""Aggregate all RGB-to-HSI validation metrics."""

from __future__ import annotations

from typing import Dict

import torch

from .mrae import mrae
from .psnr import psnr
from .rmse import rmse
from .sam import sam
from .ssim import ssim
from .utils import prepare_metric_tensors


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mrae_eps: float = 1e-6,
    ssim_window_size: int = 3,
) -> Dict[str, float]:
    pred_eval, target_eval, data_range = prepare_metric_tensors(
        pred,
        target,
        eps=mrae_eps,
    )

    return {
        "mrae": float(mrae(pred_eval, target_eval, eps=mrae_eps).item()),
        "rmse": float(rmse(pred_eval, target_eval).item()),
        "psnr": float(psnr(pred_eval, target_eval, data_range=data_range).item()),
        "sam": float(sam(pred_eval, target_eval, eps=mrae_eps, degrees=True).item()),
        "ssim": float(
            ssim(
                pred_eval,
                target_eval,
                data_range=data_range,
                window_size=ssim_window_size,
            ).item()
        ),
    }
