"""Spectral Angle Mapper for hyperspectral cubes."""

from __future__ import annotations

import torch


def sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    degrees: bool = True,
) -> torch.Tensor:
    """Average angle between predicted and target spectra at every pixel."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    if pred.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], received {pred.shape}")

    pred_spectra = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    target_spectra = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

    numerator = torch.sum(pred_spectra * target_spectra, dim=1)
    denominator = (
        torch.linalg.vector_norm(pred_spectra, dim=1)
        * torch.linalg.vector_norm(target_spectra, dim=1)
    )
    cosine = numerator / torch.clamp(denominator, min=eps)
    angles = torch.acos(torch.clamp(cosine, -1.0, 1.0))
    if degrees:
        angles = torch.rad2deg(angles)
    return angles.mean()
