#!/usr/bin/env python3
"""Diffusion Posterior Sampling model for RGB-to-HSI reconstruction.

The model follows the central idea of Chung et al., "Diffusion Posterior
Sampling for General Noisy Inverse Problems" (ICLR 2023):

1. Train an unconditional diffusion prior p(x) on clean hyperspectral cubes x.
2. Define a differentiable measurement operator A that maps HSI to RGB.
3. At inference, guide every reverse-diffusion step with the gradient of
   ||y - A(x_0_hat)||, where y is the observed RGB image.

Unlike a conditional RGB-to-HSI diffusion network, RGB is not concatenated to
or encoded by the denoiser. It only appears in the posterior/likelihood
correction during sampling.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Serializable model and diffusion configuration."""

    num_bands: int = 31
    hsi_min: float = 0.0
    hsi_max: float = 1.0

    # U-Net. The defaults are sized for 128x128 patch training and 256x256
    # batch-size-one inference on a modern GPU.
    base_channels: int = 48
    channel_mults: Tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    attention_levels: Tuple[int, ...] = (1, 2, 3)
    num_attention_heads: int = 4
    time_embedding_dim: int = 192
    dropout: float = 0.0
    group_norm_groups: int = 8

    # DDPM training process.
    diffusion_timesteps: int = 1000
    beta_schedule: str = "cosine"  # "cosine" or "linear"
    linear_beta_start: float = 1e-4
    linear_beta_end: float = 2e-2

    # DPS/DDIM inference.
    sampling_steps: int = 50
    ddim_eta: float = 0.0
    guidance_scale: float = 0.5
    normalize_guidance: bool = False
    clip_denoised: bool = True

    # Differentiable HSI -> RGB measurement operator.
    wavelengths_nm: Tuple[float, ...] = tuple(float(v) for v in range(400, 701, 10))
    trainable_srf: bool = True
    trainable_camera_gain: bool = True
    trainable_camera_gamma: bool = True
    use_camera_gamma: bool = True
    camera_gamma_init: float = 2.2

    def __post_init__(self) -> None:
        self.channel_mults = tuple(int(v) for v in self.channel_mults)
        self.attention_levels = tuple(int(v) for v in self.attention_levels)
        self.wavelengths_nm = tuple(float(v) for v in self.wavelengths_nm)

        if self.num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if len(self.wavelengths_nm) != self.num_bands:
            raise ValueError(
                "wavelengths_nm must contain exactly num_bands values; "
                f"received {len(self.wavelengths_nm)} and {self.num_bands}."
            )
        if self.hsi_max <= self.hsi_min:
            raise ValueError("hsi_max must be greater than hsi_min")
        if self.base_channels <= 0 or self.num_res_blocks <= 0:
            raise ValueError("base_channels and num_res_blocks must be positive")
        if not self.channel_mults:
            raise ValueError("channel_mults cannot be empty")
        if self.diffusion_timesteps < 2:
            raise ValueError("diffusion_timesteps must be at least 2")
        if self.sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if not 0.0 <= self.ddim_eta <= 1.0:
            raise ValueError("ddim_eta must lie in [0, 1]")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict) -> "ModelConfig":
        if not isinstance(values, dict):
            raise TypeError("ModelConfig.from_dict expects a dictionary")
        valid_names = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in valid_names}
        for tuple_name in ("channel_mults", "attention_levels", "wavelengths_nm"):
            if tuple_name in filtered:
                filtered[tuple_name] = tuple(filtered[tuple_name])
        return cls(**filtered)


# -----------------------------------------------------------------------------
# Small neural-network utilities
# -----------------------------------------------------------------------------


def _valid_group_count(channels: int, requested_groups: int) -> int:
    groups = min(int(requested_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _valid_head_count(channels: int, requested_heads: int) -> int:
    heads = min(int(requested_heads), int(channels))
    while heads > 1 and channels % heads != 0:
        heads -= 1
    return heads


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse softplus input must be positive")
    return math.log(math.expm1(value))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4:
            raise ValueError("Time embedding dimension must be at least 4")
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            timesteps = timesteps.reshape(-1)
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half, device=timesteps.device, dtype=torch.float32
        ) / max(half - 1, 1)
        frequencies = torch.exp(exponent)
        arguments = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((arguments.sin(), arguments.cos()), dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


class ResBlock(nn.Module):
    """Residual block with FiLM-like timestep modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float,
        norm_groups: int,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            _valid_group_count(in_channels, norm_groups), in_channels
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, 2 * out_channels)
        self.norm2 = nn.GroupNorm(
            _valid_group_count(out_channels, norm_groups), out_channels
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_projection(F.silu(time_embedding)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return residual + h


class SpectralChannelAttention(nn.Module):
    """Memory-efficient channel attention suitable for hyperspectral features.

    Attention is evaluated over per-head feature channels instead of over all
    spatial pixels. Its memory therefore scales linearly with image area.
    """

    def __init__(self, channels: int, requested_heads: int):
        super().__init__()
        self.channels = int(channels)
        self.heads = _valid_head_count(channels, requested_heads)
        self.head_dim = channels // self.heads

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.qkv_depthwise = nn.Conv2d(
            3 * channels,
            3 * channels,
            3,
            padding=1,
            groups=3 * channels,
            bias=False,
        )
        self.temperature = nn.Parameter(torch.ones(self.heads, 1, 1))
        self.projection = _zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_depthwise(self.qkv(self.norm(x)))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.heads, self.head_dim, h * w)
        k = k.reshape(b, self.heads, self.head_dim, h * w)
        v = v.reshape(b, self.heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = (attention * self.temperature[None]).softmax(dim=-1)
        output = torch.matmul(attention, v).reshape(b, c, h, w)
        return x + self.projection(output)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DenoiserUNet(nn.Module):
    """Unconditional epsilon-prediction U-Net for 31-band HSI cubes."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        base = config.base_channels
        time_dim = config.time_embedding_dim

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(config.num_bands, base, 3, padding=1)

        self.down_levels = nn.ModuleList()
        skip_channels: List[int] = [base]
        channels = base

        for level_index, multiplier in enumerate(config.channel_mults):
            out_channels = base * multiplier
            residual_blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                residual_blocks.append(
                    ResBlock(
                        channels,
                        out_channels,
                        time_dim,
                        config.dropout,
                        config.group_norm_groups,
                    )
                )
                channels = out_channels
                attentions.append(
                    SpectralChannelAttention(channels, config.num_attention_heads)
                    if level_index in config.attention_levels
                    else nn.Identity()
                )
                skip_channels.append(channels)

            downsample = (
                Downsample(channels)
                if level_index != len(config.channel_mults) - 1
                else nn.Identity()
            )
            if level_index != len(config.channel_mults) - 1:
                skip_channels.append(channels)

            self.down_levels.append(
                nn.ModuleDict(
                    {
                        "residual_blocks": residual_blocks,
                        "attentions": attentions,
                        "downsample": downsample,
                    }
                )
            )

        self.middle_block1 = ResBlock(
            channels,
            channels,
            time_dim,
            config.dropout,
            config.group_norm_groups,
        )
        self.middle_attention = SpectralChannelAttention(
            channels, config.num_attention_heads
        )
        self.middle_block2 = ResBlock(
            channels,
            channels,
            time_dim,
            config.dropout,
            config.group_norm_groups,
        )

        self.up_levels = nn.ModuleList()
        skip_stack = list(skip_channels)
        reversed_levels = list(reversed(range(len(config.channel_mults))))
        for level_index in reversed_levels:
            out_channels = base * config.channel_mults[level_index]
            residual_blocks = nn.ModuleList()
            attentions = nn.ModuleList()

            # One extra block consumes the skip produced by the previous level's
            # downsampling operation (or the initial input convolution at level 0).
            for _ in range(config.num_res_blocks + 1):
                skip_channels_for_block = skip_stack.pop()
                residual_blocks.append(
                    ResBlock(
                        channels + skip_channels_for_block,
                        out_channels,
                        time_dim,
                        config.dropout,
                        config.group_norm_groups,
                    )
                )
                channels = out_channels
                attentions.append(
                    SpectralChannelAttention(channels, config.num_attention_heads)
                    if level_index in config.attention_levels
                    else nn.Identity()
                )

            upsample = Upsample(channels) if level_index != 0 else nn.Identity()
            self.up_levels.append(
                nn.ModuleDict(
                    {
                        "residual_blocks": residual_blocks,
                        "attentions": attentions,
                        "upsample": upsample,
                    }
                )
            )

        if skip_stack:
            raise RuntimeError("Internal U-Net skip-channel construction error")

        self.output_norm = nn.GroupNorm(
            _valid_group_count(channels, config.group_norm_groups), channels
        )
        self.output_conv = _zero_module(
            nn.Conv2d(channels, config.num_bands, 3, padding=1)
        )

    def forward(self, noisy_hsi: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        h = self.input_conv(noisy_hsi)
        skips: List[torch.Tensor] = [h]

        for level_index, level in enumerate(self.down_levels):
            residual_blocks = level["residual_blocks"]
            attentions = level["attentions"]
            for residual_block, attention in zip(residual_blocks, attentions):
                h = residual_block(h, time_embedding)
                h = attention(h)
                skips.append(h)
            if level_index != len(self.down_levels) - 1:
                h = level["downsample"](h)
                skips.append(h)

        h = self.middle_block1(h, time_embedding)
        h = self.middle_attention(h)
        h = self.middle_block2(h, time_embedding)

        for level_index, level in enumerate(self.up_levels):
            residual_blocks = level["residual_blocks"]
            attentions = level["attentions"]
            for residual_block, attention in zip(residual_blocks, attentions):
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat((h, skip), dim=1)
                h = residual_block(h, time_embedding)
                h = attention(h)
            if level_index != len(self.up_levels) - 1:
                h = level["upsample"](h)

        if skips:
            raise RuntimeError("Internal U-Net forward skip-stack error")
        return self.output_conv(F.silu(self.output_norm(h)))


# -----------------------------------------------------------------------------
# Differentiable HSI -> RGB forward operator
# -----------------------------------------------------------------------------


class SpectralResponseOperator(nn.Module):
    """Physically constrained, differentiable camera response model.

    Each RGB response curve is non-negative and sums to one across the 31
    wavelengths. Optional positive channel gains and gamma correction absorb
    differences between linear HSI intensity and rendered/JPEG RGB values.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        wavelengths = torch.tensor(config.wavelengths_nm, dtype=torch.float32)
        self.register_buffer("wavelengths_nm", wavelengths, persistent=True)

        # Broad camera-like initialization: red, green, blue response peaks.
        centers = torch.tensor([610.0, 540.0, 460.0], dtype=torch.float32)
        widths = torch.tensor([45.0, 40.0, 35.0], dtype=torch.float32)
        response = torch.exp(
            -0.5
            * ((wavelengths[None, :] - centers[:, None]) / widths[:, None]).square()
        )
        response = response / response.sum(dim=1, keepdim=True)
        self.response_logits = nn.Parameter(
            torch.log(response.clamp_min(1e-8)),
            requires_grad=config.trainable_srf,
        )

        self.raw_gain = nn.Parameter(
            torch.full((3,), _inverse_softplus(1.0), dtype=torch.float32),
            requires_grad=config.trainable_camera_gain,
        )
        gamma_offset = max(config.camera_gamma_init - 0.5, 1e-3)
        self.raw_gamma = nn.Parameter(
            torch.full(
                (3,), _inverse_softplus(gamma_offset), dtype=torch.float32
            ),
            requires_grad=config.trainable_camera_gamma,
        )

    def response_matrix(self) -> torch.Tensor:
        """Return the current [3, 31] non-negative normalized SRF matrix."""
        return self.response_logits.softmax(dim=1)

    def gains(self) -> torch.Tensor:
        return F.softplus(self.raw_gain).clamp_min(1e-4)

    def gammas(self) -> torch.Tensor:
        return 0.5 + F.softplus(self.raw_gamma)

    def set_response_matrix(
        self,
        matrix: torch.Tensor,
        *,
        trainable: Optional[bool] = None,
    ) -> None:
        """Initialize/fix the response from a user-provided [3,31] matrix."""
        value = torch.as_tensor(matrix, dtype=self.response_logits.dtype)
        if value.shape == (self.config.num_bands, 3):
            value = value.transpose(0, 1)
        expected = (3, self.config.num_bands)
        if tuple(value.shape) != expected:
            raise ValueError(
                f"Response matrix must have shape {expected} or its transpose; "
                f"received {tuple(value.shape)}."
            )
        value = value.clamp_min(0)
        if torch.any(value.sum(dim=1) <= 0):
            raise ValueError("Every RGB response row must contain positive mass")
        value = value / value.sum(dim=1, keepdim=True)
        with torch.no_grad():
            self.response_logits.copy_(
                torch.log(value.clamp_min(1e-8)).to(self.response_logits.device)
            )
        if trainable is not None:
            self.response_logits.requires_grad_(bool(trainable))

    def smoothness_loss(self) -> torch.Tensor:
        response = self.response_matrix()
        if response.shape[1] < 3:
            return response.new_zeros(())
        second_difference = response[:, 2:] - 2.0 * response[:, 1:-1] + response[:, :-2]
        return second_difference.square().mean()

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        if hsi.ndim != 4 or hsi.shape[1] != self.config.num_bands:
            raise ValueError(
                f"Expected HSI [B,{self.config.num_bands},H,W], "
                f"received {tuple(hsi.shape)}"
            )
        response = self.response_matrix().to(dtype=hsi.dtype)
        rgb_linear = torch.einsum("bchw,rc->brhw", hsi, response)
        rgb_linear = rgb_linear * self.gains().to(hsi.dtype)[None, :, None, None]
        if self.config.use_camera_gamma:
            gamma = self.gammas().to(hsi.dtype)[None, :, None, None]
            rgb_linear = rgb_linear.clamp_min(1e-6).pow(1.0 / gamma)
        return rgb_linear


# -----------------------------------------------------------------------------
# Diffusion and DPS wrapper
# -----------------------------------------------------------------------------


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    cumulative = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5).square()
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-8, 0.999).float()


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    values = buffer.gather(0, timesteps.long())
    return values.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class DPSRGB2HSI(nn.Module):
    """Unconditional HSI DDPM prior with RGB-guided DPS reconstruction."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        self.denoiser = DenoiserUNet(self.config)
        self.camera = SpectralResponseOperator(self.config)

        if self.config.beta_schedule.lower() == "cosine":
            betas = _cosine_beta_schedule(self.config.diffusion_timesteps)
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
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alphas", alphas, persistent=True)
        self.register_buffer("alpha_bars", alpha_bars, persistent=True)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev, persistent=True)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt(), persistent=True)
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            (1.0 - alpha_bars).sqrt(),
            persistent=True,
        )
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.register_buffer(
            "posterior_variance", posterior_variance.clamp_min(1e-20), persistent=True
        )

    # -------------------------- data scaling ---------------------------------

    def normalize_hsi(self, hsi: torch.Tensor) -> torch.Tensor:
        scale = self.config.hsi_max - self.config.hsi_min
        return 2.0 * (hsi - self.config.hsi_min) / scale - 1.0

    def denormalize_hsi(self, normalized_hsi: torch.Tensor) -> torch.Tensor:
        scale = self.config.hsi_max - self.config.hsi_min
        return 0.5 * (normalized_hsi + 1.0) * scale + self.config.hsi_min

    # -------------------------- DDPM training --------------------------------

    def q_sample(
        self,
        clean_hsi_normalized: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_hsi_normalized)
        return (
            _extract(self.sqrt_alpha_bars, timesteps, clean_hsi_normalized.shape)
            * clean_hsi_normalized
            + _extract(
                self.sqrt_one_minus_alpha_bars,
                timesteps,
                clean_hsi_normalized.shape,
            )
            * noise
        )

    def predict_clean_from_noise(
        self,
        noisy_hsi: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = _extract(self.alpha_bars, timesteps, noisy_hsi.shape)
        return (
            noisy_hsi - (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt().clamp_min(1e-8)

    def forward(self, noisy_hsi: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict epsilon. RGB is intentionally not an input to the prior."""
        return self.denoiser(noisy_hsi, timesteps)

    def training_losses(
        self,
        clean_hsi: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if clean_hsi.ndim != 4 or clean_hsi.shape[1] != self.config.num_bands:
            raise ValueError(
                f"Expected clean_hsi [B,{self.config.num_bands},H,W], "
                f"received {tuple(clean_hsi.shape)}"
            )
        batch_size = clean_hsi.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.config.diffusion_timesteps,
                (batch_size,),
                device=clean_hsi.device,
            )
        if noise is None:
            noise = torch.randn_like(clean_hsi)

        clean_normalized = self.normalize_hsi(clean_hsi)
        noisy = self.q_sample(clean_normalized, timesteps, noise=noise)
        predicted_noise = self(noisy, timesteps)
        diffusion_loss = F.mse_loss(predicted_noise.float(), noise.float())
        predicted_clean_normalized = self.predict_clean_from_noise(
            noisy, timesteps, predicted_noise
        )
        if self.config.clip_denoised:
            predicted_clean_normalized = predicted_clean_normalized.clamp(-1.0, 1.0)
        predicted_clean = self.denormalize_hsi(predicted_clean_normalized)

        return {
            "diffusion_loss": diffusion_loss,
            "predicted_noise": predicted_noise,
            "noise": noise,
            "timesteps": timesteps,
            "noisy_hsi": noisy,
            "predicted_clean_hsi": predicted_clean,
        }

    # -------------------------- DPS inference --------------------------------

    @staticmethod
    def _straight_through_clamp(
        tensor: torch.Tensor, minimum: float, maximum: float
    ) -> torch.Tensor:
        clipped = tensor.clamp(minimum, maximum)
        return tensor + (clipped - tensor).detach()

    @contextmanager
    def _frozen_parameters_for_sampling(self):
        states = [(parameter, parameter.requires_grad) for parameter in self.parameters()]
        try:
            for parameter, _ in states:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, required in states:
                parameter.requires_grad_(required)

    def _sampling_schedule(self, steps: int, device: torch.device) -> torch.Tensor:
        steps = max(1, min(int(steps), self.config.diffusion_timesteps))
        values = torch.linspace(
            self.config.diffusion_timesteps - 1,
            0,
            steps,
            device=device,
        ).round().long()
        return torch.unique_consecutive(values)

    def _measurement_value_and_gradient(
        self,
        current_noisy: torch.Tensor,
        predicted_clean_normalized: torch.Tensor,
        observed_rgb: torch.Tensor,
        normalize_guidance: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted_clean_hsi = self.denormalize_hsi(predicted_clean_normalized)
        predicted_rgb = self.camera(predicted_clean_hsi)
        residual = observed_rgb - predicted_rgb

        # The official DPS implementation uses the L2 norm of the measurement
        # residual. Averaging per-sample norms keeps batches independent in scale.
        measurement_value = torch.linalg.vector_norm(
            residual.flatten(1), ord=2, dim=1
        ).mean()
        gradient = torch.autograd.grad(
            measurement_value,
            current_noisy,
            retain_graph=False,
            create_graph=False,
        )[0]

        if normalize_guidance:
            axes = tuple(range(1, gradient.ndim))
            rms = gradient.square().mean(dim=axes, keepdim=True).sqrt()
            gradient = gradient / rms.clamp_min(1e-8)
        return measurement_value.detach(), gradient

    def sample(
        self,
        rgb: torch.Tensor,
        *,
        sampling_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        ddim_eta: Optional[float] = None,
        normalize_guidance: Optional[bool] = None,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, List[float]]]]:
        """Reconstruct a hyperspectral cube from RGB using DPS.

        Args:
            rgb: Observed RGB tensor [B,3,H,W], expected in approximately [0,1].
            sampling_steps: Number of spaced DDIM reverse steps.
            guidance_scale: DPS likelihood-gradient step size. This usually needs
                tuning for a new camera response/dataset.
            ddim_eta: 0 gives deterministic DDIM transitions; >0 adds noise.
            normalize_guidance: Normalize each likelihood gradient by its RMS.
            initial_noise: Optional reproducible [B,31,H,W] starting sample.
            generator: Optional torch.Generator for sampling noise.
            return_diagnostics: Also return measurement-residual history.
        """
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB [B,3,H,W], received {tuple(rgb.shape)}")

        steps = sampling_steps or self.config.sampling_steps
        scale = self.config.guidance_scale if guidance_scale is None else guidance_scale
        eta = self.config.ddim_eta if ddim_eta is None else ddim_eta
        normalize = (
            self.config.normalize_guidance
            if normalize_guidance is None
            else normalize_guidance
        )
        if steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if scale < 0:
            raise ValueError("guidance_scale cannot be negative")
        if not 0.0 <= eta <= 1.0:
            raise ValueError("ddim_eta must lie in [0,1]")

        shape = (
            rgb.shape[0],
            self.config.num_bands,
            rgb.shape[-2],
            rgb.shape[-1],
        )
        if initial_noise is None:
            try:
                current = torch.randn(
                    shape,
                    device=rgb.device,
                    dtype=rgb.dtype,
                    generator=generator,
                )
            except TypeError:
                current = torch.randn(shape, device=rgb.device, dtype=rgb.dtype)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"initial_noise must have shape {shape}, "
                    f"received {tuple(initial_noise.shape)}"
                )
            current = initial_noise.to(device=rgb.device, dtype=rgb.dtype)

        schedule = self._sampling_schedule(int(steps), rgb.device)
        diagnostics: Dict[str, List[float]] = {"measurement_norm": []}
        was_training = self.training
        self.eval()

        with self._frozen_parameters_for_sampling():
            for index, timestep_value in enumerate(schedule.tolist()):
                next_timestep_value = (
                    int(schedule[index + 1].item())
                    if index + 1 < len(schedule)
                    else -1
                )
                batch_timesteps = torch.full(
                    (rgb.shape[0],),
                    int(timestep_value),
                    device=rgb.device,
                    dtype=torch.long,
                )

                # DPS requires backpropagation from A(x_0_hat) through x_0_hat
                # and the denoiser to the current noisy state x_t.
                with torch.enable_grad():
                    current_with_grad = current.detach().requires_grad_(True)
                    predicted_noise = self(current_with_grad, batch_timesteps)
                    predicted_clean = self.predict_clean_from_noise(
                        current_with_grad,
                        batch_timesteps,
                        predicted_noise,
                    )
                    if self.config.clip_denoised:
                        predicted_clean = self._straight_through_clamp(
                            predicted_clean, -1.0, 1.0
                        )

                    measurement_value, measurement_gradient = (
                        self._measurement_value_and_gradient(
                            current_with_grad,
                            predicted_clean,
                            rgb,
                            bool(normalize),
                        )
                    )

                alpha_t = self.alpha_bars[int(timestep_value)].to(
                    device=rgb.device, dtype=current.dtype
                )
                alpha_next = (
                    self.alpha_bars[next_timestep_value].to(
                        device=rgb.device, dtype=current.dtype
                    )
                    if next_timestep_value >= 0
                    else current.new_tensor(1.0)
                )

                sigma = eta * torch.sqrt(
                    ((1.0 - alpha_next) / (1.0 - alpha_t).clamp_min(1e-12))
                    * (1.0 - alpha_t / alpha_next.clamp_min(1e-12)).clamp_min(0.0)
                )
                direction_coefficient = torch.sqrt(
                    (1.0 - alpha_next - sigma.square()).clamp_min(0.0)
                )

                with torch.no_grad():
                    if next_timestep_value >= 0 and float(sigma.item()) > 0.0:
                        try:
                            transition_noise = torch.randn(
                                current.shape,
                                device=current.device,
                                dtype=current.dtype,
                                generator=generator,
                            )
                        except TypeError:
                            transition_noise = torch.randn_like(current)
                    else:
                        transition_noise = torch.zeros_like(current)

                    next_sample = (
                        alpha_next.sqrt() * predicted_clean.detach()
                        + direction_coefficient * predicted_noise.detach()
                        + sigma * transition_noise
                    )
                    # Posterior sampling correction, matching the official DPS
                    # pattern x_{t-1} <- x_{t-1} - zeta * grad_{x_t} ||y-A(x0_hat)||.
                    current = next_sample - float(scale) * measurement_gradient.detach()
                    diagnostics["measurement_norm"].append(float(measurement_value.item()))

        if was_training:
            self.train()
        output = self.denormalize_hsi(current).clamp(
            self.config.hsi_min, self.config.hsi_max
        )
        if return_diagnostics:
            return output, diagnostics
        return output

    def camera_regularization_loss(self) -> torch.Tensor:
        return self.camera.smoothness_loss()

    def load_fixed_srf(self, matrix: torch.Tensor) -> None:
        """Load a known [3,31] camera SRF and freeze it."""
        self.camera.set_response_matrix(matrix, trainable=False)
