#!/usr/bin/env python3
"""Train and evaluate learned-forward-operator DPS for RGB-to-HSI.

Edit the CONFIG section, then run one of:

    python train.py --mode train_forward
    python train.py --mode train_prior
    python train.py --mode eval
    python train.py --mode all

Stages
------
1. ``train_forward`` learns A_phi: 31-band HSI -> RGB.
2. ``train_prior`` freezes A_phi and trains an unconditional HSI diffusion prior.
3. ``eval`` reconstructs HSI from RGB using:

       x_init = A^T (A A^T + lambda I)^-1 (y - b)

   followed by warm-start DDIM and Diffusion Posterior Sampling.

All standard reconstruction losses and metrics are imported from the existing
``loss`` package in this repository; none are redefined here.
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
from torch.utils.data import DataLoader

from dataset.dataset_loader import ARADDataset
from dataset.random_arad_loader import load_random_arad1k_samples
from loss import (
    compute_metrics,
    l1_loss,
    mrae,
    mse_loss,
    psnr,
    reconstruction_loss,
    rmse,
    sam,
    ssim,
)
from models import DPSRGB2HSI, ModelConfig


# =============================================================================
# CONFIG
# =============================================================================

MODE = "train_forward"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
VAL_SEED = 1234

# Dataset.
DATA_ROOT = "data"
HSI_KEY = "cube"
DOWNLOAD_DATA = True
TRAIN_IMAGES = 200
TOTAL_IMAGES = 230
TRAIN_PATCH_SIZE = 256
USE_GEOMETRIC_AUGMENTATION = False
BATCH_SIZE = 2
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"

# HSI range used by the diffusion scaling and all HSI losses.
HSI_MIN = 0.0
HSI_MAX = 1.0
CLAMP_HSI = True
NUM_BANDS = 31
MRAE_EPS = 1e-6
SSIM_WINDOW_SIZE = 3

# Learned HSI -> RGB forward operator.
FORWARD_EPOCHS = 30
FORWARD_LR = 2e-3
FORWARD_WEIGHT_DECAY = 0.0
FORWARD_L1_WEIGHT = 1.0
FORWARD_MSE_WEIGHT = 0.25
FORWARD_SSIM_WEIGHT = 0.10
FORWARD_SMOOTHNESS_WEIGHT = 1e-3
FORWARD_PATIENCE = 8

# Diffusion prior architecture.
BASE_CHANNELS = 48
CHANNEL_MULTS = (1, 2, 3, 4)
NUM_RES_BLOCKS = 2
ATTENTION_LEVELS = (2, 3)
NUM_ATTENTION_HEADS = 4
TIME_EMBEDDING_DIM = 192
DROPOUT = 0.0
GROUP_NORM_GROUPS = 8

# Diffusion process.
DIFFUSION_TIMESTEPS = 1000
BETA_SCHEDULE = "cosine"
LINEAR_BETA_START = 1e-4
LINEAR_BETA_END = 2e-2

# Prior training.
PRIOR_EPOCHS = 100
PRIOR_LR = 2e-4
PRIOR_WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
USE_AMP = True
USE_EMA = True
EMA_DECAY = 0.9999
PRIOR_PATIENCE = 20
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-7

# Composite prior objective. The noise-prediction MSE remains dominant.
# Each function below comes from the repository's existing loss folder.
LAMBDA_DIFFUSION = 1.0
LAMBDA_MRAE = 0.75
LAMBDA_SAM = 0.01
LAMBDA_PSNR = 0.05
LAMBDA_SSIM = 0.05

# DPS reconstruction.
SAMPLING_STEPS = 50
DDIM_ETA = 0.0
DPS_GUIDANCE_SCALE = 0.01
NORMALIZE_DPS_GUIDANCE = True
WARM_START_STRENGTH = 0.70
SOLUTION_RIDGE = 1e-3
CLIP_DENOISED = True

# Full DPS validation is expensive. A small fixed subset is sufficient while
# training; complete evaluation is performed in --mode eval.
DPS_VALIDATE_EVERY = 5
DPS_VAL_MAX_IMAGES: Optional[int] = 2
DPS_VAL_SAMPLING_STEPS = 20

# Evaluation dataset. Set True to use the repository's fixed random ARAD1K
# loader instead of the ordinary held-out split.
EVAL_USE_RANDOM_ARAD1K = True
EVAL_RANDOM_IMAGES = 50
EVAL_RANDOM_TOTAL_IMAGES = 1000
EVAL_MAX_IMAGES: Optional[int] = None

# Checkpoints.
CHECKPOINT_DIR = Path("checkpoints")
FORWARD_BEST_PATH = CHECKPOINT_DIR / "learned_forward_operator_best.pth"
FORWARD_LATEST_PATH = CHECKPOINT_DIR / "learned_forward_operator_latest.pth"
PRIOR_BEST_PATH = CHECKPOINT_DIR / "dps_prior_best.pth"
PRIOR_LATEST_PATH = CHECKPOINT_DIR / "dps_prior_latest.pth"
PRIOR_BEST_RECON_PATH = CHECKPOINT_DIR / "dps_best_reconstruction_mrae.pth"

RESUME_FORWARD: Optional[Union[str, Path]] = None
RESUME_PRIOR: Optional[Union[str, Path]] = None
EVAL_CHECKPOINT: Optional[Union[str, Path]] = None

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ARGUMENTS AND REPRODUCIBILITY
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learned forward operator + DPS RGB-to-HSI")
    parser.add_argument(
        "--mode",
        choices=["train_forward", "train_prior", "eval", "all"],
        default=MODE,
    )
    return parser.parse_args()


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


# =============================================================================
# MODEL AND DATA
# =============================================================================


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
        warm_start_strength=WARM_START_STRENGTH,
        solution_ridge=SOLUTION_RIDGE,
        clip_denoised=CLIP_DENOISED,
    )


def build_model(device: torch.device) -> DPSRGB2HSI:
    return DPSRGB2HSI(make_model_config()).to(device)


def make_dataloaders(device: torch.device) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ARADDataset(
        root_dir=DATA_ROOT,
        train=True,
        train_images=TRAIN_IMAGES,
        total_images=TOTAL_IMAGES,
        cube_key=HSI_KEY,
        download=DOWNLOAD_DATA,
    )
    val_dataset = ARADDataset(
        root_dir=DATA_ROOT,
        train=False,
        train_images=TRAIN_IMAGES,
        total_images=TOTAL_IMAGES,
        cube_key=HSI_KEY,
        download=False,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Empty train/validation split. Check DATA_ROOT and pairing.")

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda" and PIN_MEMORY),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def make_eval_loader(device: torch.device) -> DataLoader:
    if EVAL_USE_RANDOM_ARAD1K:
        dataset, selected = load_random_arad1k_samples(
            root_dir=DATA_ROOT,
            num_samples=EVAL_RANDOM_IMAGES,
            seed=VAL_SEED,
            total_images=EVAL_RANDOM_TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )
        print(f"Random evaluation subset contains {len(selected)} samples.")
    else:
        dataset = ARADDataset(
            root_dir=DATA_ROOT,
            train=False,
            train_images=TRAIN_IMAGES,
            total_images=TOTAL_IMAGES,
            cube_key=HSI_KEY,
            download=DOWNLOAD_DATA,
        )
    return DataLoader(
        dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda" and PIN_MEMORY),
        worker_init_fn=seed_worker if NUM_WORKERS > 0 else None,
    )


# =============================================================================
# BATCH AND TRAINING UTILITIES
# =============================================================================


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        rgb = batch.get("rgb", batch.get("lq"))
        hsi = batch.get("hsi", batch.get("gt"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        rgb, hsi = batch[0], batch[1]
    else:
        raise TypeError("Expected batch as dict or (rgb, hsi) tuple/list")

    if not torch.is_tensor(rgb) or not torch.is_tensor(hsi):
        raise TypeError("RGB and HSI must be tensors")
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB [B,3,H,W], got {tuple(rgb.shape)}")
    if hsi.ndim != 4 or hsi.shape[1] != NUM_BANDS:
        raise ValueError(f"Expected HSI [B,{NUM_BANDS},H,W], got {tuple(hsi.shape)}")
    return rgb, hsi


def prepare_hsi(hsi: torch.Tensor) -> torch.Tensor:
    return hsi.clamp(HSI_MIN, HSI_MAX) if CLAMP_HSI else hsi


def random_crop_and_augment(
    rgb: torch.Tensor,
    hsi: torch.Tensor,
    patch_size: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = hsi.shape[-2:]
    if patch_size is not None and patch_size > 0:
        if patch_size > height or patch_size > width:
            raise ValueError(
                f"TRAIN_PATCH_SIZE={patch_size} exceeds image size {height}x{width}"
            )
        rgb_patches = []
        hsi_patches = []
        for sample in range(hsi.shape[0]):
            top = random.randint(0, height - patch_size)
            left = random.randint(0, width - patch_size)
            rgb_patches.append(rgb[sample : sample + 1, :, top : top + patch_size, left : left + patch_size])
            hsi_patches.append(hsi[sample : sample + 1, :, top : top + patch_size, left : left + patch_size])
        rgb = torch.cat(rgb_patches, dim=0)
        hsi = torch.cat(hsi_patches, dim=0)

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


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)


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


class ExponentialMovingAverage:
    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict, device: torch.device) -> None:
        self.decay = float(state["decay"])
        self.shadow = {
            name: tensor.detach().to(device).clone()
            for name, tensor in state["shadow"].items()
        }

    @contextmanager
    def average_parameters(self, module: torch.nn.Module):
        backup = {}
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if name in self.shadow:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in module.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])

    @torch.no_grad()
    def copy_to(self, module: torch.nn.Module) -> None:
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name])


# =============================================================================
# CHECKPOINTS
# =============================================================================


def save_forward_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: DPSRGB2HSI,
    optimizer: torch.optim.Optimizer,
    best_val_l1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "forward",
            "epoch": epoch,
            "model_config": model.config.to_dict(),
            "forward_operator": model.forward_operator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_l1": best_val_l1,
        },
        path,
    )


def save_prior_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: DPSRGB2HSI,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    ema: Optional[ExponentialMovingAverage],
    best_val_loss: float,
    best_recon_mrae: float,
    bad_epochs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "prior",
        "epoch": epoch,
        "model_config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_recon_mrae": best_recon_mrae,
        "bad_epochs": bad_epochs,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Union[str, Path], device: torch.device) -> Dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: {path}")
    return checkpoint


def load_forward_operator(
    model: DPSRGB2HSI,
    path: Union[str, Path],
    device: torch.device,
) -> Dict:
    checkpoint = load_checkpoint(path, device)
    if "forward_operator" in checkpoint:
        state = checkpoint["forward_operator"]
    elif "model" in checkpoint:
        prefix = "forward_operator."
        state = {
            key[len(prefix) :]: value
            for key, value in checkpoint["model"].items()
            if key.startswith(prefix)
        }
    else:
        raise KeyError("Checkpoint contains neither 'forward_operator' nor 'model'")
    model.forward_operator.load_state_dict(state, strict=True)
    return checkpoint


# =============================================================================
# STAGE 1: LEARN FORWARD OPERATOR A_phi
# =============================================================================


def forward_objective(
    model: DPSRGB2HSI,
    hsi: torch.Tensor,
    rgb: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predicted_rgb = model.project_hsi_to_rgb(hsi)

    # Imported from the existing loss package.
    rgb_l1 = l1_loss(predicted_rgb, rgb)
    rgb_mse = mse_loss(predicted_rgb, rgb)
    rgb_ssim = ssim(
        predicted_rgb,
        rgb,
        data_range=1.0,
        window_size=SSIM_WINDOW_SIZE,
    )
    smoothness = model.forward_operator.smoothness_regularizer()
    total = (
        FORWARD_L1_WEIGHT * rgb_l1
        + FORWARD_MSE_WEIGHT * rgb_mse
        + FORWARD_SSIM_WEIGHT * (1.0 - rgb_ssim)
        + FORWARD_SMOOTHNESS_WEIGHT * smoothness
    )
    return total, {
        "l1": rgb_l1,
        "mse": rgb_mse,
        "ssim": rgb_ssim,
        "smooth": smoothness,
    }


@torch.no_grad()
def validate_forward(
    model: DPSRGB2HSI,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "l1": 0.0, "mse": 0.0, "rmse": 0.0, "psnr": 0.0, "ssim": 0.0}
    count = 0
    for batch in loader:
        rgb, hsi = unpack_batch(batch)
        rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
        hsi = prepare_hsi(hsi).to(device, non_blocking=(device.type == "cuda"))
        predicted = model.project_hsi_to_rgb(hsi)
        total, parts = forward_objective(model, hsi, rgb)
        batch_size = rgb.shape[0]
        totals["total"] += float(total.item()) * batch_size
        totals["l1"] += float(parts["l1"].item()) * batch_size
        totals["mse"] += float(parts["mse"].item()) * batch_size
        totals["rmse"] += float(rmse(predicted, rgb).item()) * batch_size
        totals["psnr"] += float(psnr(predicted, rgb, data_range=1.0).item()) * batch_size
        totals["ssim"] += float(parts["ssim"].item()) * batch_size
        count += batch_size
    if count == 0:
        raise RuntimeError("Forward validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def train_forward_operator() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    train_loader, val_loader = make_dataloaders(device)
    model = build_model(device)
    model.freeze_prior()
    model.unfreeze_forward_operator()

    optimizer = torch.optim.AdamW(
        model.forward_operator_parameters(),
        lr=FORWARD_LR,
        weight_decay=FORWARD_WEIGHT_DECAY,
    )
    start_epoch = 1
    best_val_l1 = math.inf
    bad_epochs = 0

    if RESUME_FORWARD is not None:
        checkpoint = load_forward_operator(model, RESUME_FORWARD, device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_l1 = float(checkpoint.get("best_val_l1", math.inf))

    print("========== STAGE 1: LEARN HSI -> RGB FORWARD OPERATOR ==========")
    print(f"Device: {device}")
    print(f"Trainable forward parameters: {count_parameters(model.forward_operator_parameters()):,}")

    for epoch in range(start_epoch, FORWARD_EPOCHS + 1):
        model.train()
        model.denoiser.eval()
        running = {"total": 0.0, "l1": 0.0, "mse": 0.0, "ssim": 0.0, "smooth": 0.0}
        count = 0

        for batch in train_loader:
            rgb, hsi = unpack_batch(batch)
            rgb, hsi = random_crop_and_augment(rgb, hsi, TRAIN_PATCH_SIZE)
            rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
            hsi = prepare_hsi(hsi).to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            total, parts = forward_objective(model, hsi, rgb)
            total.backward()
            optimizer.step()

            batch_size = rgb.shape[0]
            running["total"] += float(total.item()) * batch_size
            for key in ("l1", "mse", "ssim", "smooth"):
                running[key] += float(parts[key].item()) * batch_size
            count += batch_size

        train_stats = {key: value / max(count, 1) for key, value in running.items()}
        val_stats = validate_forward(model, val_loader, device)

        print(
            f"Forward E{epoch:03d}/{FORWARD_EPOCHS} "
            f"| Train Total {train_stats['total']:.6f} "
            f"| L1 {train_stats['l1']:.6f} "
            f"| MSE {train_stats['mse']:.6f} "
            f"| SSIM {train_stats['ssim']:.6f} "
            f"| Smooth {train_stats['smooth']:.6f} "
            f"| Val L1 {val_stats['l1']:.6f} "
            f"| Val RMSE {val_stats['rmse']:.6f} "
            f"| Val PSNR {val_stats['psnr']:.4f} "
            f"| Val SSIM {val_stats['ssim']:.6f}"
        )

        save_forward_checkpoint(
            FORWARD_LATEST_PATH,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_val_l1=min(best_val_l1, val_stats["l1"]),
        )

        if val_stats["l1"] < best_val_l1:
            best_val_l1 = val_stats["l1"]
            bad_epochs = 0
            save_forward_checkpoint(
                FORWARD_BEST_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_l1=best_val_l1,
            )
            print(f"Saved best forward operator: Val RGB L1 {best_val_l1:.6f}")
        else:
            bad_epochs += 1
            if bad_epochs >= FORWARD_PATIENCE:
                print(f"Forward-stage early stopping after {bad_epochs} non-improving epochs.")
                break

    matrix = model.forward_operator.matrix().detach().cpu()
    print("Learned A matrix shape:", tuple(matrix.shape))
    print("Forward checkpoint:", FORWARD_BEST_PATH)


# =============================================================================
# STAGE 2: TRAIN UNCONDITIONAL HSI DIFFUSION PRIOR
# =============================================================================


def prior_objective(
    model: DPSRGB2HSI,
    hsi: torch.Tensor,
    *,
    timesteps: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    outputs = model.training_predictions(hsi, timesteps=timesteps, noise=noise)
    predicted_noise = outputs["predicted_noise"]
    target_noise = outputs["noise"]
    predicted_clean = outputs["predicted_clean_hsi"]

    # Every standard loss below is imported from the existing loss package.
    diffusion_loss = mse_loss(predicted_noise, target_noise)
    mrae_loss = mrae(predicted_clean, hsi, eps=MRAE_EPS)
    sam_loss = sam(predicted_clean, hsi, eps=MRAE_EPS, degrees=False)
    psnr_value = psnr(predicted_clean, hsi, data_range=HSI_MAX - HSI_MIN)
    psnr_loss = torch.pow(predicted_clean.new_tensor(10.0), -psnr_value / 10.0)
    ssim_value = ssim(
        predicted_clean,
        hsi,
        data_range=HSI_MAX - HSI_MIN,
        window_size=SSIM_WINDOW_SIZE,
    )
    ssim_loss = 1.0 - ssim_value

    total = (
        LAMBDA_DIFFUSION * diffusion_loss
        + LAMBDA_MRAE * mrae_loss
        + LAMBDA_SAM * sam_loss
        + LAMBDA_PSNR * psnr_loss
        + LAMBDA_SSIM * ssim_loss
    )
    return total, {
        "diffusion": diffusion_loss,
        "mrae": mrae_loss,
        "sam": sam_loss,
        "psnr": psnr_value,
        "psnr_loss": psnr_loss,
        "ssim": ssim_value,
        "ssim_loss": ssim_loss,
    }


@torch.no_grad()
def validate_prior(
    model: DPSRGB2HSI,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "total": 0.0,
        "diffusion": 0.0,
        "mrae": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    count = 0
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(VAL_SEED)

    for batch in loader:
        _, hsi = unpack_batch(batch)
        hsi = prepare_hsi(hsi).to(device, non_blocking=(device.type == "cuda"))
        batch_size = hsi.shape[0]
        timesteps = torch.randint(
            0,
            DIFFUSION_TIMESTEPS,
            (batch_size,),
            device=device,
            generator=generator,
        )
        noise = torch.randn(hsi.shape, device=device, dtype=hsi.dtype, generator=generator)
        total, parts = prior_objective(model, hsi, timesteps=timesteps, noise=noise)
        totals["total"] += float(total.item()) * batch_size
        for key in ("diffusion", "mrae", "sam", "psnr", "ssim"):
            totals[key] += float(parts[key].item()) * batch_size
        count += batch_size

    if count == 0:
        raise RuntimeError("Prior validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def validate_reconstruction(
    model: DPSRGB2HSI,
    loader: DataLoader,
    device: torch.device,
    *,
    max_images: Optional[int],
    sampling_steps: int,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "mrae": 0.0,
        "rmse": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "rgb_l1": 0.0,
        "init_mrae": 0.0,
    }
    count = 0
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(VAL_SEED)

    for batch in loader:
        rgb, hsi = unpack_batch(batch)
        rgb = rgb.to(device, non_blocking=(device.type == "cuda"))
        hsi = prepare_hsi(hsi).to(device, non_blocking=(device.type == "cuda"))

        with torch.no_grad():
            initial_hsi = model.minimum_norm_solution(rgb, ridge=SOLUTION_RIDGE)
        reconstructed = model.sample(
            rgb,
            sampling_steps=sampling_steps,
            guidance_scale=DPS_GUIDANCE_SCALE,
            warm_start_strength=WARM_START_STRENGTH,
            solution_ridge=SOLUTION_RIDGE,
            ddim_eta=DDIM_ETA,
            normalize_guidance=NORMALIZE_DPS_GUIDANCE,
            generator=generator,
        )
        with torch.no_grad():
            reconstructed_rgb = model.project_hsi_to_rgb(reconstructed)

        for sample in range(rgb.shape[0]):
            pred = reconstructed[sample : sample + 1]
            target = hsi[sample : sample + 1]
            metrics = compute_metrics(
                pred,
                target,
                mrae_eps=MRAE_EPS,
                ssim_window_size=SSIM_WINDOW_SIZE,
            )
            sample_loss = reconstruction_loss(
                pred,
                target,
                loss_type="mrae",
                mrae_eps=MRAE_EPS,
            )
            totals["loss"] += float(sample_loss.item())
            for key in ("mrae", "rmse", "sam", "psnr", "ssim"):
                totals[key] += float(metrics[key])
            totals["rgb_l1"] += float(
                l1_loss(
                    reconstructed_rgb[sample : sample + 1],
                    rgb[sample : sample + 1],
                ).item()
            )
            totals["init_mrae"] += float(
                mrae(
                    initial_hsi[sample : sample + 1],
                    target,
                    eps=MRAE_EPS,
                ).item()
            )
            count += 1
            if max_images is not None and count >= max_images:
                break
        if max_images is not None and count >= max_images:
            break

    if count == 0:
        raise RuntimeError("Reconstruction validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def train_diffusion_prior() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    if not FORWARD_BEST_PATH.exists() and RESUME_PRIOR is None:
        raise FileNotFoundError(
            f"Forward checkpoint not found: {FORWARD_BEST_PATH}\n"
            "Run: python train.py --mode train_forward"
        )

    train_loader, val_loader = make_dataloaders(device)
    model = build_model(device)
    optimizer = torch.optim.AdamW(
        model.prior_parameters(),
        lr=PRIOR_LR,
        weight_decay=PRIOR_WEIGHT_DECAY,
        betas=(0.9, 0.99),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )
    amp_enabled = USE_AMP and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)
    ema = ExponentialMovingAverage(model.denoiser, EMA_DECAY) if USE_EMA else None

    start_epoch = 1
    best_val_loss = math.inf
    best_recon_mrae = math.inf
    bad_epochs = 0

    if RESUME_PRIOR is not None:
        checkpoint = load_checkpoint(RESUME_PRIOR, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"], device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        best_recon_mrae = float(checkpoint.get("best_recon_mrae", math.inf))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
    else:
        load_forward_operator(model, FORWARD_BEST_PATH, device)

    model.freeze_forward_operator()
    model.unfreeze_prior()

    print("========== STAGE 2: TRAIN UNCONDITIONAL HSI DIFFUSION PRIOR ==========")
    print(f"Device: {device}")
    print(f"Trainable prior parameters: {count_parameters(model.prior_parameters()):,}")
    print(f"Loaded and frozen forward operator: {FORWARD_BEST_PATH}")

    for epoch in range(start_epoch, PRIOR_EPOCHS + 1):
        model.train()
        model.forward_operator.eval()
        running = {
            "total": 0.0,
            "diffusion": 0.0,
            "mrae": 0.0,
            "sam": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
        }
        count = 0

        for batch in train_loader:
            rgb, hsi = unpack_batch(batch)
            _, hsi = random_crop_and_augment(rgb, hsi, TRAIN_PATCH_SIZE)
            hsi = prepare_hsi(hsi).to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(amp_enabled):
                total, parts = prior_objective(model, hsi)

            scaler.scale(total).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.prior_parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model.denoiser)

            batch_size = hsi.shape[0]
            running["total"] += float(total.item()) * batch_size
            for key in ("diffusion", "mrae", "sam", "psnr", "ssim"):
                running[key] += float(parts[key].item()) * batch_size
            count += batch_size

        train_stats = {key: value / max(count, 1) for key, value in running.items()}

        if ema is not None:
            with ema.average_parameters(model.denoiser):
                val_stats = validate_prior(model, val_loader, device)
        else:
            val_stats = validate_prior(model, val_loader, device)

        scheduler.step(val_stats["total"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Prior E{epoch:03d}/{PRIOR_EPOCHS} "
            f"| Train Total {train_stats['total']:.6f} "
            f"| Diff {train_stats['diffusion']:.6f} "
            f"| MRAE {train_stats['mrae']:.6f} "
            f"| SAM(rad) {train_stats['sam']:.6f} "
            f"| PSNR {train_stats['psnr']:.4f} "
            f"| SSIM {train_stats['ssim']:.6f} "
            f"| Val Total {val_stats['total']:.6f} "
            f"| Val Diff {val_stats['diffusion']:.6f} "
            f"| Val MRAE {val_stats['mrae']:.6f} "
            f"| Val SAM(rad) {val_stats['sam']:.6f} "
            f"| Val PSNR {val_stats['psnr']:.4f} "
            f"| Val SSIM {val_stats['ssim']:.6f} "
            f"| LR {current_lr:.2e}"
        )

        if val_stats["total"] < best_val_loss:
            best_val_loss = val_stats["total"]
            bad_epochs = 0
            save_prior_checkpoint(
                PRIOR_BEST_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                best_val_loss=best_val_loss,
                best_recon_mrae=best_recon_mrae,
                bad_epochs=bad_epochs,
            )
            print(f"Saved best prior: Val objective {best_val_loss:.6f}")
        else:
            bad_epochs += 1

        if epoch % DPS_VALIDATE_EVERY == 0 or epoch == PRIOR_EPOCHS:
            if ema is not None:
                with ema.average_parameters(model.denoiser):
                    recon = validate_reconstruction(
                        model,
                        val_loader,
                        device,
                        max_images=DPS_VAL_MAX_IMAGES,
                        sampling_steps=DPS_VAL_SAMPLING_STEPS,
                    )
            else:
                recon = validate_reconstruction(
                    model,
                    val_loader,
                    device,
                    max_images=DPS_VAL_MAX_IMAGES,
                    sampling_steps=DPS_VAL_SAMPLING_STEPS,
                )
            print(
                f"DPS Val | Init MRAE {recon['init_mrae']:.6f} "
                f"| MRAE {recon['mrae']:.6f} "
                f"| RMSE {recon['rmse']:.6f} "
                f"| SAM {recon['sam']:.4f} "
                f"| PSNR {recon['psnr']:.4f} "
                f"| SSIM {recon['ssim']:.6f} "
                f"| RGB L1 {recon['rgb_l1']:.6f}"
            )
            if recon["mrae"] < best_recon_mrae:
                best_recon_mrae = recon["mrae"]
                save_prior_checkpoint(
                    PRIOR_BEST_RECON_PATH,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ema=ema,
                    best_val_loss=best_val_loss,
                    best_recon_mrae=best_recon_mrae,
                    bad_epochs=bad_epochs,
                )
                print(f"Saved best DPS reconstruction: MRAE {best_recon_mrae:.6f}")

        save_prior_checkpoint(
            PRIOR_LATEST_PATH,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            best_val_loss=best_val_loss,
            best_recon_mrae=best_recon_mrae,
            bad_epochs=bad_epochs,
        )

        if bad_epochs >= PRIOR_PATIENCE:
            print(f"Prior-stage early stopping after {bad_epochs} non-improving epochs.")
            break


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate() -> None:
    set_seed(SEED)
    device = torch.device(DEVICE)
    default_path = PRIOR_BEST_RECON_PATH if PRIOR_BEST_RECON_PATH.exists() else PRIOR_BEST_PATH
    checkpoint_path = Path(EVAL_CHECKPOINT) if EVAL_CHECKPOINT is not None else default_path
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = DPSRGB2HSI(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    if "ema" in checkpoint:
        ema = ExponentialMovingAverage(model.denoiser, checkpoint["ema"]["decay"])
        ema.load_state_dict(checkpoint["ema"], device)
        ema.copy_to(model.denoiser)
        print("Using EMA denoiser parameters.")

    model.freeze_forward_operator()
    model.eval()
    loader = make_eval_loader(device)
    results = validate_reconstruction(
        model,
        loader,
        device,
        max_images=EVAL_MAX_IMAGES,
        sampling_steps=SAMPLING_STEPS,
    )

    print("========== FINAL RGB -> HSI DPS EVALUATION ==========")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Minimum-norm initialization MRAE: {results['init_mrae']:.6f}")
    print(
        f"MRAE {results['mrae']:.6f} "
        f"| RMSE {results['rmse']:.6f} "
        f"| SAM {results['sam']:.4f} "
        f"| PSNR {results['psnr']:.4f} "
        f"| SSIM {results['ssim']:.6f} "
        f"| RGB L1 {results['rgb_l1']:.6f}"
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if args.mode == "train_forward":
        train_forward_operator()
    elif args.mode == "train_prior":
        train_diffusion_prior()
    elif args.mode == "eval":
        evaluate()
    elif args.mode == "all":
        train_forward_operator()
        train_diffusion_prior()
        evaluate()
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
