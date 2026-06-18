"""Losses used to supervise the Stage-2 compact DiffIR prior."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def prior_l1_loss(
    student_prior: torch.Tensor,
    teacher_prior: torch.Tensor,
) -> torch.Tensor:
    """Absolute matching loss between predicted and oracle compact priors."""
    if student_prior.shape != teacher_prior.shape:
        raise ValueError(
            "student_prior and teacher_prior must have the same shape, "
            f"got {student_prior.shape} and {teacher_prior.shape}"
        )
    return F.l1_loss(student_prior, teacher_prior.detach())


def prior_kd_loss(
    student_prior: torch.Tensor,
    teacher_prior: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    """KL-divergence knowledge-distillation loss for compact priors.

    The teacher is detached so gradients are applied only to Stage 2.
    """
    if student_prior.shape != teacher_prior.shape:
        raise ValueError(
            "student_prior and teacher_prior must have the same shape, "
            f"got {student_prior.shape} and {teacher_prior.shape}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    student_log_prob = F.log_softmax(student_prior / temperature, dim=1)
    teacher_prob = F.softmax(teacher_prior.detach() / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
