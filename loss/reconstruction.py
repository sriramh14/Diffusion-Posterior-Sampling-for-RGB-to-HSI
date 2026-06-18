"""Select the reconstruction objective from one common function."""

from __future__ import annotations

import torch

from .l1 import l1_loss
from .mrae import mrae
from .mse import mse_loss


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mrae",
    mrae_eps: float = 1e-6,
) -> torch.Tensor:
    """Compute MRAE, L1, or MSE according to ``loss_type``."""
    name = loss_type.lower().strip()
    if name == "mrae":
        return mrae(pred, target, eps=mrae_eps)
    if name in {"l1", "mae"}:
        return l1_loss(pred, target)
    if name in {"mse", "l2"}:
        return mse_loss(pred, target)
    raise ValueError(
        "loss_type must be one of: 'mrae', 'l1'/'mae', or 'mse'/'l2'; "
        f"received {loss_type!r}"
    )
