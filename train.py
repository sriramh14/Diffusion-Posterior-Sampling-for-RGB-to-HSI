#!/usr/bin/env python3
"""Single-file training/evaluation entry point for DPS RGB-to-HSI.

Edit the CONFIG section and run:

    python train.py --mode train
    python train.py --mode eval

There are no training stages. The same optimization run learns:

1. an unconditional 31-band HSI diffusion prior, and
2. a differentiable HSI-to-RGB camera response used by DPS at inference.

The RGB image is deliberately not passed to the diffusion U-Net. During
reconstruction it conditions reverse diffusion through the measurement-gradient
term from Diffusion Posterior Sampling.
"""

from __future__ import annotations

import argparse
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataset_loader import ARADDataset
from dataset.random_arad_loader import load_random_arad1k_samples
from loss import compute_metrics, reconstruction_loss
from models.dps_rgb2hsi_model import DPSRGB2HSI, ModelConfig


# ==================================================
# CONFIG
# ==================================================

MODE = "train"                 # "train" or "eval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_SEED = 1234

# Dataset configuration.
DATA_ROOT = "data"
HSI_KEY = "cube"
DOWNLOAD_DATA = True
TRAIN_IMAGES = 2
TOTAL_IMAGES = 4
EVAL_RANDOM_IMAGES = 50
EVAL_RANDOM_TOTAL_IMAGES = 1000

# Patch training keeps direct 31-channel diffusion practical. Validation and
# evaluation still reconstruct the full 256x256 images from the loaders.
TRAIN_PATCH_SIZE = 256
USE_GEOMETRIC_AUGMENTATION = False
BATCH_SIZE = 4
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"

NUM_EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
USE_AMP = True

# ARAD reflectance cubes are normally in [0,1]. The diffusion data scaling and
# camera model below assume this range. Disable clamping only after checking the
# actual min/max values in your .mat files and changing HSI_MIN/HSI_MAX.
HSI_MIN = 0.0
HSI_MAX = 1.0
CLAMP_HSI_TO_CONFIG_RANGE = True

# Training objective.
# Noise-prediction MSE is the main DDPM objective. The small clean-HSI and
# spectral-gradient terms stabilize a diffusion prior trained on a small set.
LAMBDA_DIFFUSION = 1.0
LAMBDA_X0_L1 = 0.10
LAMBDA_SPECTRAL_GRAD = 0.05
LAMBDA_CAMERA_RGB = 0.10
LAMBDA_SRF_SMOOTH = 1e-3

# Validation reconstruction loss and metrics.
RECONSTRUCTION_LOSS = "mrae"
MRAE_EPS = 1e-6

# Architecture.
NUM_BANDS = 31
BASE_CHANNELS = 48
CHANNEL_MULTS = (1, 2, 3, 4)
NUM_RES_BLOCKS = 2
ATTENTION_LEVELS = (1, 2, 3)
NUM_ATTENTION_HEADS = 4
TIME_EMBEDDING_DIM = 192
DROPOUT = 0.0
GROUP_NORM_GROUPS = 8

# DDPM prior training.
DIFFUSION_TIMESTEPS = 1000
BETA_SCHEDULE = "cosine"       # "cosine" or "linear"
LINEAR_BETA_START = 1e-4
LINEAR_BETA_END = 2e-2

# DPS reconstruction. Guidance scale is the most important value to tune.
# Try values such as 0.1, 0.25, 0.5, 1.0, and 2.0 on the validation set.
SAMPLING_STEPS = 50
DDIM_ETA = 0.0
DPS_GUIDANCE_SCALE = 0.5
NORMALIZE_DPS_GUIDANCE = False
CLIP_DENOISED = True

# Camera response. If a measured 3x31 spectral response matrix is available,
# set FIXED_SRF_PATH to a .npy file; otherwise a smooth positive SRF is learned
# jointly from paired GT-HSI/RGB data in this same training run.
FIXED_SRF_PATH: Optional[Union[str, Path]] = None
TRAINABLE_SRF = FIXED_SRF_PATH is None
TRAINABLE_CAMERA_GAIN = True
TRAINABLE_CAMERA_GAMMA = True
USE_CAMERA_GAMMA = True
CAMERA_GAMMA_INIT = 2.2

# Full DPS validation is considerably more expensive than a feed-forward model.
# Set VAL_MAX_IMAGES=None for the complete validation split.
VAL_MAX_IMAGES: Optional[int] = 10
VAL_SAMPLING_STEPS = 30
VAL_GUIDANCE_SCALE = DPS_GUIDANCE_SCALE

# Optimization control.
EARLY_STOPPING_PATIENCE = 20
LR_PATIENCE = 4
LR_FACTOR = 0.5
MIN_LR = 1e-7
USE_EMA = True
EMA_DECAY = 0.9999

CHECKPOINT_DIR = Path("checkpoints")
BEST_PATH = CHECKPOINT_DIR / "dps_rgb2hsi_best.pth"
BEST_LOSS_PATH = CHECKPOINT_DIR / "dps_rgb2hsi_best_loss.pth"
LATEST_PATH = CHECKPOINT_DIR / "dps_rgb2hsi_latest.pth"
RESUME_CHECKPOINT: Optional[Union[str, Path]] = None
EVAL_CHECKPOINT: Optional[Union[str, Path]] = None

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# COMMAND LINE OVERRIDE
# --------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate DPS RGB-to-HSI")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default=MODE,
        help="Run mode: train or eval.",
    )
    return parser.parse_args()


args = parse_args()
MODE = args.mode


# ==================================================
# REPRODUCIBILITY
# ==================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ==================================================
# MODEL CONFIGURATION
# ==================================================


def make_model_config() -> ModelConfig:
    return ModelConfig(
        num_bands=NUM_BANDS,
        hsi_min=HSI_MIN,
        hsi_max=HSI_MAX,
        base_channels=BASE_CHANNELS,
        channel_mults=CHANNEL_MULTS,
        num_res_blocks=NUM_RES_BLOCKS,
        attention_levels=ATTENTION_LEVELS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        time_embedding_dim=TIME_EMBEDDING_DIM,
        dropout=DROPOUT,
        group_norm_groups=GROUP_NORM_GROUPS,
        diffusion_timesteps=DIFFUSION_TIMESTEPS,
        beta_schedule=BETA_SCHEDULE,
        linear_beta_start=LINEAR_BETA_START,
        linear_beta_end=LINEAR_BETA_END,
        sampling_steps=SAMPLING_STEPS,
        ddim_eta=DDIM_ETA,
        guidance_scale=DPS_GUIDANCE_SCALE,
        normalize_guidance=NORMALIZE_DPS_GUIDANCE,
        clip_denoised=CLIP_DENOISED,
        trainable_srf=TRAINABLE_SRF,
        trainable_camera_gain=TRAINABLE_CAMERA_GAIN,
        trainable_camera_gamma=TRAINABLE_CAMERA_GAMMA,
        use_camera_gamma=USE_CAMERA_GAMMA,
        camera_gamma_init=CAMERA_GAMMA_INIT,
    )


def build_model(config: ModelConfig, device: torch.device) -> DPSRGB2HSI:
    model = DPSRGB2HSI(config).to(device)
    if FIXED_SRF_PATH is not None:
        path = Path(FIXED_SRF_PATH)
        if not path.exists():
            raise FileNotFoundError(f"FIXED_SRF_PATH not found: {path}")
        matrix = torch.from_numpy(np.load(path)).float()
        model.load_fixed_srf(matrix)
        print(f"Loaded and froze measured camera SRF: {path}")
    return model


# ==================================================
# SMALL UTILITIES
# ==================================================


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Any, Optional[torch.Tensor]]:
    """Normalize tuple/list/dict datasets to rgb, hsi, name, orig_hw."""
    name = None
    orig_hw = None

    if isinstance(batch, dict):
        rgb = batch.get("rgb", batch.get("lq"))
        hsi = batch.get("hsi", batch.get("gt"))
        name = batch.get("name", batch.get("filename"))
        orig_hw = batch.get("orig_hw", batch.get("original_hw"))
        if rgb is None or hsi is None:
            raise KeyError(
                "Dictionary batch must contain ('rgb','hsi') or ('lq','gt'). "
                f"Available keys: {list(batch.keys())}"
            )
    elif isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("List/tuple batch must contain at least [rgb, hsi].")
        rgb, hsi = batch[0], batch[1]
        if len(batch) >= 3:
            name = batch[2]
        if len(batch) >= 4:
            orig_hw = batch[3]
    else:
        raise TypeError(
            "Expected a dict, list, or tuple batch, but received "
            f"{type(batch).__name__}."
        )

    if not torch.is_tensor(rgb) or not torch.is_tensor(hsi):
        raise TypeError("RGB and HSI must be tensors after DataLoader collation")
    if rgb.ndim != 4 or hsi.ndim != 4:
        raise ValueError(
            "Expected rgb=[B,3,H,W] and hsi=[B,31,H,W], got "
            f"rgb={tuple(rgb.shape)}, hsi={tuple(hsi.shape)}"
        )
    if rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB to have 3 channels, got {rgb.shape[1]}")
    if hsi.shape[1] != NUM_BANDS:
        raise ValueError(f"Expected HSI to have {NUM_BANDS} bands, got {hsi.shape[1]}")
    return rgb, hsi, name, orig_hw


def prepare_hsi(hsi: torch.Tensor) -> torch.Tensor:
    if CLAMP_HSI_TO_CONFIG_RANGE:
        return hsi.clamp(HSI_MIN, HSI_MAX)
    return hsi


def random_crop_and_augment(
    rgb: torch.Tensor,
    hsi: torch.Tensor,
    patch_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if patch_size is not None and patch_size > 0:
        height, width = hsi.shape[-2:]
        if patch_size > height or patch_size > width:
            raise ValueError(
                f"TRAIN_PATCH_SIZE={patch_size} exceeds sample size {height}x{width}"
            )
        cropped_rgb = []
        cropped_hsi = []
        for index in range(hsi.shape[0]):
            top = random.randint(0, height - patch_size)
            left = random.randint(0, width - patch_size)
            cropped_rgb.append(
                rgb[index : index + 1, :, top : top + patch_size, left : left + patch_size]
            )
            cropped_hsi.append(
                hsi[index : index + 1, :, top : top + patch_size, left : left + patch_size]
            )
        rgb = torch.cat(cropped_rgb, dim=0)
        hsi = torch.cat(cropped_hsi, dim=0)

    if USE_GEOMETRIC_AUGMENTATION:
        if random.random() < 0.5:
            rgb = torch.flip(rgb, dims=(-1,))
            hsi = torch.flip(hsi, dims=(-1,))
        if random.random() < 0.5:
            rgb = torch.flip(rgb, dims=(-2,))
            hsi = torch.flip(hsi, dims=(-2,))
        rotations = random.randint(0, 3)
        if rotations:
            rgb = torch.rot90(rgb, rotations, dims=(-2, -1))
            hsi = torch.rot90(hsi, rotations, dims=(-2, -1))
    return rgb.contiguous(), hsi.contiguous()


def make_orig_hw_tensor(orig_hw: Optional[Any], hsi: torch.Tensor) -> torch.Tensor:
    batch_size = hsi.shape[0]
    if orig_hw is None:
        return torch.tensor(
            [[hsi.shape[-2], hsi.shape[-1]]] * batch_size,
            dtype=torch.long,
        )
    value = orig_hw.detach().cpu() if torch.is_tensor(orig_hw) else torch.as_tensor(orig_hw)
    if value.ndim == 1 and value.numel() == 2:
        value = value.view(1, 2).repeat(batch_size, 1)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"orig_hw must have shape [B,2], got {tuple(value.shape)}")
    if value.shape[0] == 1 and batch_size > 1:
        value = value.repeat(batch_size, 1)
    if value.shape[0] != batch_size:
        raise ValueError("orig_hw batch dimension does not match HSI batch")
    return value.long()


def crop_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    orig_hw: torch.Tensor,
    sample_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    height = int(orig_hw[sample_index, 0].item())
    width = int(orig_hw[sample_index, 1].item())
    return (
        pred[sample_index : sample_index + 1, :, :height, :width],
        target[sample_index : sample_index + 1, :, :height, :width],
    )


def spectral_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] < 2:
        return pred.new_zeros(())
    pred_gradient = pred[:, 1:] - pred[:, :-1]
    target_gradient = target[:, 1:] - target[:, :-1]
    return F.l1_loss(pred_gradient, target_gradient)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# ==================================================
# EMA
# ==================================================


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie in (0,1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    parameter.detach(), alpha=1.0 - self.decay
                )

    def state_dict(self) -> Dict:
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict, device: torch.device) -> None:
        self.decay = float(state["decay"])
        loaded = state["shadow"]
        missing = set(self.shadow) - set(loaded)
        if missing:
            raise KeyError(f"EMA checkpoint is missing parameters: {sorted(missing)[:5]}")
        self.shadow = {
            name: loaded[name].to(device=device, dtype=self.shadow[name].dtype)
            for name in self.shadow
        }

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        swapped = []
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in self.shadow:
                    continue
                current = parameter.detach().clone()
                parameter.copy_(self.shadow[name])
                swapped.append((name, parameter, current))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter, current in swapped:
                    parameter.copy_(current)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name])


# ==================================================
# DATA
# ==================================================


def make_dataloaders(device: torch.device) -> Tuple[Optional[DataLoader], DataLoader]:
    set_seed(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader: Optional[DataLoader] = None
    if MODE == "train":
        train_dataset = ARADDataset(
            root_dir=DATA_ROOT,
            train=True,
            train_images=TRAIN_IMAGES,
            total_images=TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )
        if len(train_dataset) == 0:
            raise RuntimeError("Training dataset is empty. Check DATA_ROOT and pairing.")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda" and PIN_MEMORY),
            worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
            generator=generator,
            drop_last=False,
        )

    if MODE == "eval":
        val_dataset, _ = load_random_arad1k_samples(
            root_dir=DATA_ROOT,
            num_samples=EVAL_RANDOM_IMAGES,
            seed=VAL_SEED,
            total_images=EVAL_RANDOM_TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )
    else:
        val_dataset = ARADDataset(
            root_dir=DATA_ROOT,
            train=False,
            train_images=TRAIN_IMAGES,
            total_images=TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=False,
        )
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty. Check the split and pairing.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda" and PIN_MEMORY),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader


# ==================================================
# CHECKPOINTS
# ==================================================


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: DPSRGB2HSI,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
    ema: Optional[ExponentialMovingAverage],
    config: ModelConfig,
    best_val_mrae: float,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "model_config": config.to_dict(),
        "best_val_mrae": best_val_mrae,
        "best_val_loss": best_val_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Union[str, Path], device: torch.device) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a complete checkpoint dictionary with key 'model'.")
    return checkpoint


# ==================================================
# VALIDATION
# ==================================================


@torch.no_grad()
def validate(
    model: DPSRGB2HSI,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_images: Optional[int],
    sampling_steps: int,
    guidance_scale: float,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "psnr": 0.0,
        "sam": 0.0,
        "ssim": 0.0,
        "rgb_l1": 0.0,
    }
    count = 0

    try:
        eval_generator = torch.Generator(device=device)
    except TypeError:
        eval_generator = torch.Generator()
    eval_generator.manual_seed(VAL_SEED)

    for batch in val_loader:
        rgb, hsi, _, orig_hw = unpack_batch(batch)
        orig_hw_tensor = make_orig_hw_tensor(orig_hw, hsi)
        rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
        hsi = prepare_hsi(hsi).to(device, non_blocking=(device.type == "cuda"))

        pred_hsi = model.sample(
            rgb,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale,
            generator=eval_generator,
        )
        pred_rgb = model.camera(pred_hsi)

        for sample_index in range(rgb.shape[0]):
            sample_pred, sample_hsi = crop_sample(
                pred_hsi, hsi, orig_hw_tensor, sample_index
            )
            sample_loss = reconstruction_loss(
                sample_pred,
                sample_hsi,
                loss_type=RECONSTRUCTION_LOSS,
                mrae_eps=MRAE_EPS,
            )
            sample_metrics = compute_metrics(
                sample_pred, sample_hsi, mrae_eps=MRAE_EPS
            )
            totals["loss"] += float(sample_loss.item())
            for metric_name in ("mrae", "rmse", "psnr", "sam", "ssim"):
                totals[metric_name] += float(sample_metrics[metric_name])
            totals["rgb_l1"] += float(
                F.l1_loss(
                    pred_rgb[sample_index : sample_index + 1],
                    rgb[sample_index : sample_index + 1],
                ).item()
            )
            count += 1
            if max_images is not None and count >= max_images:
                break
        if max_images is not None and count >= max_images:
            break

    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return {name: value / count for name, value in totals.items()}


# ==================================================
# TRAINING
# ==================================================


def train() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader = make_dataloaders(device)
    if train_loader is None:
        raise RuntimeError("Training requested but train_loader is None")

    config = make_model_config()
    model = build_model(config, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )
    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    ema = ExponentialMovingAverage(model, EMA_DECAY) if USE_EMA else None

    start_epoch = 1
    best_val_mrae = math.inf
    best_val_loss = math.inf
    epochs_without_improvement = 0

    if RESUME_CHECKPOINT is not None:
        resume = load_checkpoint(RESUME_CHECKPOINT, device)
        resume_config = ModelConfig.from_dict(resume["model_config"])
        if resume_config.to_dict() != config.to_dict():
            raise ValueError("Resume checkpoint architecture differs from current CONFIG")
        model.load_state_dict(resume["model"], strict=True)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            lr_scheduler.load_state_dict(resume["scheduler"])
        if ema is not None and "ema" in resume:
            ema.load_state_dict(resume["ema"], device)
        start_epoch = int(resume.get("epoch", 0)) + 1
        best_val_mrae = float(resume.get("best_val_mrae", math.inf))
        best_val_loss = float(resume.get("best_val_loss", math.inf))
        epochs_without_improvement = int(resume.get("epochs_without_improvement", 0))
        print(f"Resumed from epoch {start_epoch}")

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Model configuration: {config.to_dict()}")
    print("Training is single-stage; RGB is used by the camera loss and DPS sampler.")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running = {
            "total": 0.0,
            "diffusion": 0.0,
            "x0_l1": 0.0,
            "spectral_grad": 0.0,
            "camera_rgb": 0.0,
            "srf_smooth": 0.0,
        }
        train_count = 0

        for batch in train_loader:
            rgb, hsi, _, _ = unpack_batch(batch)
            rgb, hsi = random_crop_and_augment(rgb, hsi, TRAIN_PATCH_SIZE)
            rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
            hsi = prepare_hsi(hsi).to(device, non_blocking=(device.type == "cuda"))
            batch_size = rgb.shape[0]
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled):
                diffusion_outputs = model.training_losses(hsi)
                diffusion_loss = diffusion_outputs["diffusion_loss"]
                predicted_clean = diffusion_outputs["predicted_clean_hsi"]
                x0_l1 = F.l1_loss(predicted_clean, hsi)
                spectral_grad = spectral_gradient_loss(predicted_clean, hsi)
                camera_rgb = F.l1_loss(model.camera(hsi), rgb)
                srf_smooth = model.camera_regularization_loss()

                total_loss = (
                    LAMBDA_DIFFUSION * diffusion_loss
                    + LAMBDA_X0_L1 * x0_l1
                    + LAMBDA_SPECTRAL_GRAD * spectral_grad
                    + LAMBDA_CAMERA_RGB * camera_rgb
                    + LAMBDA_SRF_SMOOTH * srf_smooth
                )

            scaler.scale(total_loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=GRAD_CLIP_NORM,
                )
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            values = {
                "total": total_loss,
                "diffusion": diffusion_loss,
                "x0_l1": x0_l1,
                "spectral_grad": spectral_grad,
                "camera_rgb": camera_rgb,
                "srf_smooth": srf_smooth,
            }
            for name, value in values.items():
                running[name] += float(value.item()) * batch_size
            train_count += batch_size

        train_stats = {
            name: value / max(train_count, 1) for name, value in running.items()
        }

        if ema is not None:
            with ema.average_parameters(model):
                val_results = validate(
                    model,
                    val_loader,
                    device,
                    max_images=VAL_MAX_IMAGES,
                    sampling_steps=VAL_SAMPLING_STEPS,
                    guidance_scale=VAL_GUIDANCE_SCALE,
                )
        else:
            val_results = validate(
                model,
                val_loader,
                device,
                max_images=VAL_MAX_IMAGES,
                sampling_steps=VAL_SAMPLING_STEPS,
                guidance_scale=VAL_GUIDANCE_SCALE,
            )

        lr_scheduler.step(val_results["mrae"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} "
            f"| Train Total {train_stats['total']:.6f} "
            f"| Diffusion {train_stats['diffusion']:.6f} "
            f"| X0 L1 {train_stats['x0_l1']:.6f} "
            f"| Spectral Grad {train_stats['spectral_grad']:.6f} "
            f"| Camera RGB {train_stats['camera_rgb']:.6f} "
            f"| Val Loss {val_results['loss']:.6f} "
            f"| Val MRAE {val_results['mrae']:.6f} "
            f"| Val RMSE {val_results['rmse']:.6f} "
            f"| Val SAM {val_results['sam']:.4f} "
            f"| Val PSNR {val_results['psnr']:.4f} "
            f"| Val SSIM {val_results['ssim']:.6f} "
            f"| Val RGB L1 {val_results['rgb_l1']:.6f} "
            f"| LR {current_lr:.2e}"
        )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            save_checkpoint(
                BEST_LOSS_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                ema=ema,
                config=config,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )

        if val_results["mrae"] < best_val_mrae:
            best_val_mrae = val_results["mrae"]
            epochs_without_improvement = 0
            save_checkpoint(
                BEST_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                ema=ema,
                config=config,
                best_val_mrae=best_val_mrae,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
            )
            print(f"Saved best model (Val MRAE: {best_val_mrae:.6f})")
        else:
            epochs_without_improvement += 1
            print(
                "No validation MRAE improvement for "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs"
            )

        save_checkpoint(
            LATEST_PATH,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            ema=ema,
            config=config,
            best_val_mrae=best_val_mrae,
            best_val_loss=best_val_loss,
            epochs_without_improvement=epochs_without_improvement,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping. Best validation MRAE: {best_val_mrae:.6f}")
            break


# ==================================================
# EVALUATION
# ==================================================


def evaluate() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    _, val_loader = make_dataloaders(device)
    selected_path = Path(EVAL_CHECKPOINT) if EVAL_CHECKPOINT is not None else BEST_PATH
    checkpoint = load_checkpoint(selected_path, device)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = DPSRGB2HSI(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    if "ema" in checkpoint:
        ema = ExponentialMovingAverage(model, decay=float(checkpoint["ema"]["decay"]))
        ema.load_state_dict(checkpoint["ema"], device)
        ema.copy_to(model)
        print("Using EMA parameters for evaluation.")

    results = validate(
        model,
        val_loader,
        device,
        max_images=None,
        sampling_steps=SAMPLING_STEPS,
        guidance_scale=DPS_GUIDANCE_SCALE,
    )
    print(f"Evaluated checkpoint: {selected_path}")
    print(f"Model configuration: {config.to_dict()}")
    print(
        f"MRAE {results['mrae']:.6f} "
        f"| RMSE {results['rmse']:.6f} "
        f"| SAM {results['sam']:.4f} "
        f"| PSNR {results['psnr']:.4f} "
        f"| SSIM {results['ssim']:.6f} "
        f"| RGB L1 {results['rgb_l1']:.6f}"
    )


# ==================================================
# MAIN
# ==================================================


def main() -> None:
    if MODE == "train":
        train()
    elif MODE == "eval":
        evaluate()
    else:
        raise ValueError("MODE must be 'train' or 'eval'")


if __name__ == "__main__":
    main()
