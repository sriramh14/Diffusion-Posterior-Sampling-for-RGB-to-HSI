"""Diffusion Posterior Sampling model for RGB-to-HSI reconstruction.

The model separates the inverse problem into two learned components:

1. ``LearnedHSIToRGBOperator`` learns the forward map A_phi: HSI -> RGB.
2. ``DiffusionUNet`` learns an unconditional prior over 31-band HSI cubes.

At inference, the learned forward operator provides a Tikhonov minimum-norm
solution, and DPS refines that solution while enforcing RGB consistency.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    num_bands: int = 31
    hsi_min: float = 0.0
    hsi_max: float = 1.0

    base_channels: int = 48
    channel_mults: Tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    attention_levels: Tuple[int, ...] = (2, 3)
    num_attention_heads: int = 4
    time_embedding_dim: int = 192
    dropout: float = 0.0
    group_norm_groups: int = 8

    diffusion_timesteps: int = 1000
    beta_schedule: str = "cosine"
    linear_beta_start: float = 1e-4
    linear_beta_end: float = 2e-2

    sampling_steps: int = 50
    ddim_eta: float = 0.0
    guidance_scale: float = 0.5
    normalize_guidance: bool = True
    warm_start_strength: float = 0.70
    solution_ridge: float = 1e-3
    clip_denoised: bool = True

    def __post_init__(self) -> None:
        self.channel_mults = tuple(int(v) for v in self.channel_mults)
        self.attention_levels = tuple(int(v) for v in self.attention_levels)
        if self.num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if self.hsi_max <= self.hsi_min:
            raise ValueError("hsi_max must be greater than hsi_min")
        if len(self.channel_mults) == 0:
            raise ValueError("channel_mults cannot be empty")
        if self.num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive")
        if self.diffusion_timesteps < 2:
            raise ValueError("diffusion_timesteps must be at least 2")
        if not 0.0 <= self.warm_start_strength <= 1.0:
            raise ValueError("warm_start_strength must lie in [0, 1]")
        if self.solution_ridge <= 0:
            raise ValueError("solution_ridge must be positive")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict) -> "ModelConfig":
        valid = {item.name for item in fields(cls)}
        clean = {key: value for key, value in values.items() if key in valid}
        if "channel_mults" in clean:
            clean["channel_mults"] = tuple(clean["channel_mults"])
        if "attention_levels" in clean:
            clean["attention_levels"] = tuple(clean["attention_levels"])
        return cls(**clean)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _group_count(channels: int, requested: int) -> int:
    groups = min(int(requested), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    values = buffer.gather(0, timesteps.long())
    return values.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


def _randn(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    return torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value.clamp_min(1e-6)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5).square()
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-8, 0.999).float()


# -----------------------------------------------------------------------------
# Learned forward operator A_phi: HSI -> RGB
# -----------------------------------------------------------------------------


class LearnedHSIToRGBOperator(nn.Module):
    """Positive, spectrally smooth, affine 1x1 HSI-to-RGB projection.

    The spectral responses are positive and normalized over wavelength. A
    learnable positive gain allows each RGB channel to have an independent
    scale. This keeps the forward map interpretable while still data-driven.
    """

    def __init__(self, num_bands: int = 31) -> None:
        super().__init__()
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        self.num_bands = int(num_bands)

        wavelengths = torch.linspace(400.0, 700.0, self.num_bands)
        centers = torch.tensor([610.0, 545.0, 460.0]).view(3, 1)
        widths = torch.tensor([48.0, 42.0, 38.0]).view(3, 1)
        response = torch.exp(-0.5 * ((wavelengths.view(1, -1) - centers) / widths).square())
        response = response / response.sum(dim=1, keepdim=True)

        self.raw_response = nn.Parameter(_inverse_softplus(response))
        self.raw_gain = nn.Parameter(_inverse_softplus(torch.ones(3)))
        self.bias = nn.Parameter(torch.zeros(3))

    def normalized_response(self) -> torch.Tensor:
        response = F.softplus(self.raw_response) + 1e-8
        return response / response.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def matrix(self) -> torch.Tensor:
        gain = F.softplus(self.raw_gain).view(3, 1)
        return gain * self.normalized_response()

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        if hsi.ndim != 4 or hsi.shape[1] != self.num_bands:
            raise ValueError(
                f"Expected HSI [B,{self.num_bands},H,W], received {tuple(hsi.shape)}"
            )
        weight = self.matrix().to(dtype=hsi.dtype).unsqueeze(-1).unsqueeze(-1)
        bias = self.bias.to(dtype=hsi.dtype)
        return F.conv2d(hsi, weight, bias)

    def smoothness_regularizer(self) -> torch.Tensor:
        response = self.normalized_response()
        if self.num_bands < 3:
            return response.new_zeros(())
        second_difference = response[:, 2:] - 2.0 * response[:, 1:-1] + response[:, :-2]
        return second_difference.square().mean()

    @torch.no_grad()
    def set_matrix(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        if weight.shape != (3, self.num_bands):
            raise ValueError(
                f"weight must have shape (3,{self.num_bands}), got {tuple(weight.shape)}"
            )
        positive = weight.detach().to(self.raw_response).clamp_min(1e-8)
        gain = positive.sum(dim=1).clamp_min(1e-8)
        response = positive / gain[:, None]
        self.raw_response.copy_(_inverse_softplus(response))
        self.raw_gain.copy_(_inverse_softplus(gain))
        if bias is not None:
            if bias.shape != (3,):
                raise ValueError(f"bias must have shape (3,), got {tuple(bias.shape)}")
            self.bias.copy_(bias.detach().to(self.bias))

    def minimum_norm_solution(
        self,
        rgb: torch.Tensor,
        ridge: float = 1e-3,
    ) -> torch.Tensor:
        """Compute A^T(AA^T + lambda I)^-1(y-b) per image pixel."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], received {tuple(rgb.shape)}")
        if ridge <= 0:
            raise ValueError("ridge must be positive")

        matrix = self.matrix().to(device=rgb.device, dtype=rgb.dtype)
        identity = torch.eye(3, device=rgb.device, dtype=rgb.dtype)
        gram = matrix @ matrix.transpose(0, 1) + float(ridge) * identity
        back_projector = matrix.transpose(0, 1) @ torch.linalg.inv(gram)
        centered = rgb - self.bias.to(rgb).view(1, 3, 1, 1)
        return F.conv2d(centered, back_projector.unsqueeze(-1).unsqueeze(-1))

    def tikhonov_correction(
        self,
        prior_hsi: torch.Tensor,
        rgb: torch.Tensor,
        ridge: float = 1e-3,
    ) -> torch.Tensor:
        """Apply x + A^T(AA^T+lambda I)^-1(y-Ax)."""
        if prior_hsi.ndim != 4 or prior_hsi.shape[1] != self.num_bands:
            raise ValueError(
                f"Expected prior_hsi [B,{self.num_bands},H,W], got {tuple(prior_hsi.shape)}"
            )
        if prior_hsi.shape[0] != rgb.shape[0] or prior_hsi.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("prior_hsi and rgb must share batch and spatial dimensions")

        matrix = self.matrix().to(device=rgb.device, dtype=rgb.dtype)
        identity = torch.eye(3, device=rgb.device, dtype=rgb.dtype)
        gram = matrix @ matrix.transpose(0, 1) + float(ridge) * identity
        back_projector = matrix.transpose(0, 1) @ torch.linalg.inv(gram)
        residual = rgb - self(prior_hsi)
        correction = F.conv2d(residual, back_projector.unsqueeze(-1).unsqueeze(-1))
        return prior_hsi + correction


# Compatibility alias for older imports.
LinearHSIToRGBOperator = LearnedHSIToRGBOperator


# -----------------------------------------------------------------------------
# Diffusion prior U-Net
# -----------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        if half == 0:
            return timesteps.float().unsqueeze(1)
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale
        )
        phases = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((phases.sin(), phases.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return embedding


class TimeResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float,
        groups: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class LinearSpatialAttention(nn.Module):
    """Memory-efficient spatial attention with linear complexity in H*W."""

    def __init__(self, channels: int, heads: int, groups: int) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.heads = int(heads)
        self.dim_head = channels // heads
        self.norm = nn.GroupNorm(_group_count(channels, groups), channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [
            tensor.reshape(batch, self.heads, self.dim_head, height * width)
            for tensor in qkv
        ]
        q = q.softmax(dim=2)
        k = k.softmax(dim=3)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        attended = torch.einsum("bhde,bhdn->bhen", context, q)
        attended = attended.reshape(batch, channels, height, width)
        return x + self.to_out(attended)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class DiffusionUNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        self.input_conv = nn.Conv2d(config.num_bands, base, 3, padding=1)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base),
            nn.Linear(base, config.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
        )

        self.down_levels = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        skip_channels = []
        current = base

        for level_index, multiplier in enumerate(config.channel_mults):
            target = base * multiplier
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                blocks.append(
                    TimeResidualBlock(
                        current,
                        target,
                        config.time_embedding_dim,
                        config.dropout,
                        config.group_norm_groups,
                    )
                )
                current = target
                attentions.append(
                    LinearSpatialAttention(
                        current,
                        config.num_attention_heads,
                        config.group_norm_groups,
                    )
                    if level_index in config.attention_levels
                    else nn.Identity()
                )
                skip_channels.append(current)
            self.down_levels.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions}))
            if level_index < len(config.channel_mults) - 1:
                self.downsamplers.append(Downsample(current))
            else:
                self.downsamplers.append(nn.Identity())

        self.mid_block1 = TimeResidualBlock(
            current,
            current,
            config.time_embedding_dim,
            config.dropout,
            config.group_norm_groups,
        )
        self.mid_attention = LinearSpatialAttention(
            current,
            config.num_attention_heads,
            config.group_norm_groups,
        )
        self.mid_block2 = TimeResidualBlock(
            current,
            current,
            config.time_embedding_dim,
            config.dropout,
            config.group_norm_groups,
        )

        self.up_levels = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        remaining_skips = list(skip_channels)
        for level_index in reversed(range(len(config.channel_mults))):
            target = base * config.channel_mults[level_index]
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                skip = remaining_skips.pop()
                blocks.append(
                    TimeResidualBlock(
                        current + skip,
                        target,
                        config.time_embedding_dim,
                        config.dropout,
                        config.group_norm_groups,
                    )
                )
                current = target
                attentions.append(
                    LinearSpatialAttention(
                        current,
                        config.num_attention_heads,
                        config.group_norm_groups,
                    )
                    if level_index in config.attention_levels
                    else nn.Identity()
                )
            self.up_levels.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions}))
            self.upsamplers.append(Upsample(current) if level_index > 0 else nn.Identity())

        if remaining_skips:
            raise RuntimeError("Internal U-Net skip-channel construction error")

        self.output_norm = nn.GroupNorm(_group_count(current, config.group_norm_groups), current)
        self.output_conv = nn.Conv2d(current, config.num_bands, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], received {tuple(x.shape)}")
        required_divisor = 2 ** (len(self.config.channel_mults) - 1)
        if x.shape[-2] % required_divisor != 0 or x.shape[-1] % required_divisor != 0:
            raise ValueError(
                f"Spatial dimensions must be divisible by {required_divisor}; got {x.shape[-2:]}"
            )
        time_embedding = self.time_mlp(timesteps)
        hidden = self.input_conv(x)
        skips = []

        for level, downsample in zip(self.down_levels, self.downsamplers):
            for block, attention in zip(level["blocks"], level["attentions"]):
                hidden = block(hidden, time_embedding)
                hidden = attention(hidden)
                skips.append(hidden)
            hidden = downsample(hidden)

        hidden = self.mid_block1(hidden, time_embedding)
        hidden = self.mid_attention(hidden)
        hidden = self.mid_block2(hidden, time_embedding)

        for level, upsample in zip(self.up_levels, self.upsamplers):
            for block, attention in zip(level["blocks"], level["attentions"]):
                skip = skips.pop()
                if hidden.shape[-2:] != skip.shape[-2:]:
                    hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
                hidden = torch.cat((hidden, skip), dim=1)
                hidden = block(hidden, time_embedding)
                hidden = attention(hidden)
            hidden = upsample(hidden)

        if skips:
            raise RuntimeError("Internal U-Net skip stack was not fully consumed")
        return self.output_conv(F.silu(self.output_norm(hidden)))


# -----------------------------------------------------------------------------
# Complete DPS model
# -----------------------------------------------------------------------------


class DPSRGB2HSI(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.forward_operator = LearnedHSIToRGBOperator(self.config.num_bands)
        self.denoiser = DiffusionUNet(self.config)

        if self.config.beta_schedule.lower() == "cosine":
            betas = cosine_beta_schedule(self.config.diffusion_timesteps)
        elif self.config.beta_schedule.lower() == "linear":
            betas = torch.linspace(
                self.config.linear_beta_start,
                self.config.linear_beta_end,
                self.config.diffusion_timesteps,
                dtype=torch.float32,
            )
        else:
            raise ValueError("beta_schedule must be 'cosine' or 'linear'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    # ----- parameter groups -------------------------------------------------

    def forward_operator_parameters(self) -> Iterable[nn.Parameter]:
        return self.forward_operator.parameters()

    def prior_parameters(self) -> Iterable[nn.Parameter]:
        return self.denoiser.parameters()

    def freeze_forward_operator(self) -> None:
        self.forward_operator.eval()
        for parameter in self.forward_operator.parameters():
            parameter.requires_grad_(False)

    def unfreeze_forward_operator(self) -> None:
        for parameter in self.forward_operator.parameters():
            parameter.requires_grad_(True)

    def freeze_prior(self) -> None:
        self.denoiser.eval()
        for parameter in self.denoiser.parameters():
            parameter.requires_grad_(False)

    def unfreeze_prior(self) -> None:
        for parameter in self.denoiser.parameters():
            parameter.requires_grad_(True)

    # ----- data scaling -----------------------------------------------------

    def to_diffusion_space(self, hsi: torch.Tensor) -> torch.Tensor:
        scale = self.config.hsi_max - self.config.hsi_min
        return 2.0 * (hsi - self.config.hsi_min) / scale - 1.0

    def from_diffusion_space(self, value: torch.Tensor) -> torch.Tensor:
        scale = self.config.hsi_max - self.config.hsi_min
        return (value + 1.0) * 0.5 * scale + self.config.hsi_min

    # ----- forward operator -------------------------------------------------

    def project_hsi_to_rgb(self, hsi: torch.Tensor) -> torch.Tensor:
        return self.forward_operator(hsi)

    def minimum_norm_solution(
        self,
        rgb: torch.Tensor,
        ridge: Optional[float] = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        solution = self.forward_operator.minimum_norm_solution(
            rgb,
            ridge=self.config.solution_ridge if ridge is None else ridge,
        )
        if clamp:
            solution = solution.clamp(self.config.hsi_min, self.config.hsi_max)
        return solution

    def correct_solution(
        self,
        prior_hsi: torch.Tensor,
        rgb: torch.Tensor,
        ridge: Optional[float] = None,
        clamp: bool = True,
    ) -> torch.Tensor:
        corrected = self.forward_operator.tikhonov_correction(
            prior_hsi,
            rgb,
            ridge=self.config.solution_ridge if ridge is None else ridge,
        )
        if clamp:
            corrected = corrected.clamp(self.config.hsi_min, self.config.hsi_max)
        return corrected

    # ----- diffusion training ----------------------------------------------

    def forward(self, noisy_hsi: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.denoiser(noisy_hsi, timesteps)

    def q_sample(
        self,
        clean_diffusion: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_diffusion)
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, clean_diffusion.shape) * clean_diffusion
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean_diffusion.shape) * noise
        )

    def predict_clean_from_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = _extract(self.alphas_cumprod, timesteps, noisy.shape)
        clean = (noisy - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        return clean.clamp(-1.0, 1.0) if self.config.clip_denoised else clean

    def training_predictions(
        self,
        clean_hsi: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch = clean_hsi.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.config.diffusion_timesteps,
                (batch,),
                device=clean_hsi.device,
            )
        if noise is None:
            noise = torch.randn_like(clean_hsi)

        clean_diffusion = self.to_diffusion_space(clean_hsi)
        noisy = self.q_sample(clean_diffusion, timesteps, noise)
        predicted_noise = self.denoiser(noisy, timesteps)
        predicted_clean_diffusion = self.predict_clean_from_noise(
            noisy,
            timesteps,
            predicted_noise,
        )
        predicted_clean_hsi = self.from_diffusion_space(predicted_clean_diffusion)
        return {
            "timesteps": timesteps,
            "noise": noise,
            "noisy_hsi": noisy,
            "predicted_noise": predicted_noise,
            "predicted_clean_hsi": predicted_clean_hsi,
        }

    # Backward-compatible name used by an earlier training script.
    def training_losses(self, clean_hsi: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.training_predictions(clean_hsi)
        outputs["diffusion_loss"] = F.mse_loss(outputs["predicted_noise"], outputs["noise"])
        return outputs

    # ----- DPS reconstruction ----------------------------------------------

    def _sampling_schedule(self, start_timestep: int, steps: int, device: torch.device) -> torch.Tensor:
        steps = max(2, min(int(steps), start_timestep + 1))
        schedule = torch.linspace(start_timestep, 0, steps, device=device).round().long()
        schedule = torch.unique_consecutive(schedule)
        if schedule[-1].item() != 0:
            schedule = torch.cat((schedule, schedule.new_zeros(1)))
        return schedule

    def sample(
        self,
        rgb: torch.Tensor,
        *,
        sampling_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        warm_start_strength: Optional[float] = None,
        solution_ridge: Optional[float] = None,
        ddim_eta: Optional[float] = None,
        normalize_guidance: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Reconstruct HSI from RGB with warm-start DDIM plus DPS guidance."""
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], received {tuple(rgb.shape)}")

        steps = self.config.sampling_steps if sampling_steps is None else int(sampling_steps)
        guidance = self.config.guidance_scale if guidance_scale is None else float(guidance_scale)
        strength = (
            self.config.warm_start_strength
            if warm_start_strength is None
            else float(warm_start_strength)
        )
        ridge = self.config.solution_ridge if solution_ridge is None else float(solution_ridge)
        eta = self.config.ddim_eta if ddim_eta is None else float(ddim_eta)
        normalize = (
            self.config.normalize_guidance
            if normalize_guidance is None
            else bool(normalize_guidance)
        )
        if not 0.0 <= strength <= 1.0:
            raise ValueError("warm_start_strength must lie in [0,1]")

        # Closed-form solution from the learned forward operator.
        with torch.no_grad():
            initial_hsi = self.minimum_norm_solution(rgb, ridge=ridge, clamp=True)
            initial = self.to_diffusion_space(initial_hsi)
            start_timestep = max(
                1,
                min(
                    self.config.diffusion_timesteps - 1,
                    int(round(strength * (self.config.diffusion_timesteps - 1))),
                ),
            )
            start_t = torch.full(
                (rgb.shape[0],),
                start_timestep,
                device=rgb.device,
                dtype=torch.long,
            )
            initial_noise = _randn(
                initial.shape,
                device=initial.device,
                dtype=initial.dtype,
                generator=generator,
            )
            current = self.q_sample(initial, start_t, initial_noise)
            schedule = self._sampling_schedule(start_timestep, steps, rgb.device)

        denoiser_was_training = self.denoiser.training
        operator_was_training = self.forward_operator.training
        self.denoiser.eval()
        self.forward_operator.eval()

        try:
            for index, timestep_value in enumerate(schedule):
                timestep = int(timestep_value.item())
                previous_timestep = (
                    int(schedule[index + 1].item()) if index + 1 < len(schedule) else -1
                )
                batch_t = torch.full(
                    (rgb.shape[0],),
                    timestep,
                    device=rgb.device,
                    dtype=torch.long,
                )

                with torch.enable_grad():
                    current_for_grad = current.detach().requires_grad_(guidance != 0.0)
                    predicted_noise = self.denoiser(current_for_grad, batch_t)
                    predicted_clean = self.predict_clean_from_noise(
                        current_for_grad,
                        batch_t,
                        predicted_noise,
                    )

                    if guidance != 0.0:
                        predicted_hsi = self.from_diffusion_space(predicted_clean)
                        predicted_rgb = self.forward_operator(predicted_hsi)
                        residual = predicted_rgb - rgb
                        measurement_objective = 0.5 * residual.square().sum()
                        measurement_gradient = torch.autograd.grad(
                            measurement_objective,
                            current_for_grad,
                            retain_graph=False,
                            create_graph=False,
                        )[0]
                    else:
                        residual = None
                        measurement_gradient = torch.zeros_like(current_for_grad)

                with torch.no_grad():
                    alpha_t = self.alphas_cumprod[timestep].to(current)
                    alpha_previous = (
                        self.alphas_cumprod[previous_timestep].to(current)
                        if previous_timestep >= 0
                        else current.new_tensor(1.0)
                    )
                    sigma = eta * torch.sqrt(
                        ((1.0 - alpha_previous) / (1.0 - alpha_t)).clamp_min(0.0)
                        * (1.0 - alpha_t / alpha_previous).clamp_min(0.0)
                    )
                    direction_coefficient = torch.sqrt(
                        (1.0 - alpha_previous - sigma.square()).clamp_min(0.0)
                    )
                    next_sample = (
                        torch.sqrt(alpha_previous) * predicted_clean.detach()
                        + direction_coefficient * predicted_noise.detach()
                    )
                    if previous_timestep >= 0 and float(sigma.item()) > 0.0:
                        next_sample = next_sample + sigma * _randn(
                            current.shape,
                            device=current.device,
                            dtype=current.dtype,
                            generator=generator,
                        )

                    if guidance != 0.0:
                        if normalize:
                            gradient_rms = measurement_gradient.square().mean(
                                dim=(1, 2, 3), keepdim=True
                            ).sqrt().clamp_min(1e-8)
                            measurement_gradient = measurement_gradient / gradient_rms
                        next_sample = next_sample - guidance * measurement_gradient
                    current = next_sample
        finally:
            self.denoiser.train(denoiser_was_training)
            self.forward_operator.train(operator_was_training)

        reconstructed = self.from_diffusion_space(current)
        return reconstructed.clamp(self.config.hsi_min, self.config.hsi_max)
