import os
import random
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from huggingface_hub import list_repo_files, hf_hub_download


class RandomARAD1KDataset(Dataset):
    """Dataset containing only a fixed random selection of ARAD1K pairs."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        selected_samples: List[Dict],
        cube_key: str = "cube",
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.pairs = pairs
        self.selected_samples = selected_samples
        self.cube_key = cube_key
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hsi_path, rgb_path = self.pairs[idx]

        mat = sio.loadmat(hsi_path)
        if self.cube_key not in mat:
            raise KeyError(
                f"Key '{self.cube_key}' was not found in {hsi_path}. "
                f"Available keys: {list(mat.keys())}"
            )

        hsi = mat[self.cube_key].astype(np.float32)
        hsi = np.transpose(hsi, (2, 0, 1))
        hsi = torch.from_numpy(hsi).float()
        hsi = F.interpolate(
            hsi.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        with Image.open(rgb_path) as image:
            rgb = image.convert("RGB")
            rgb = np.asarray(rgb, dtype=np.float32) / 255.0

        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.from_numpy(rgb).float()
        rgb = F.interpolate(
            rgb.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return rgb, hsi


def _sample_remote_pairs(
    total_images: int,
    num_samples: int,
    seed: int,
):
    """List the repository, pair filenames, then sample before downloading."""
    repo_files = list_repo_files(
        "mhmdjouni/arad_hsdb",
        repo_type="dataset",
    )

    hsi_files = sorted(
        path
        for path in repo_files
        if path.endswith(".mat") and "NTIRE2020_Train_Spectral" in path
    )[:total_images]

    rgb_files = sorted(
        path
        for path in repo_files
        if path.endswith(".jpg") and "NTIRE2020_Train_RealWorld" in path
    )[:total_images]

    rgb_lookup = {
        os.path.basename(path).replace("_RealWorld.jpg", ""): path
        for path in rgb_files
    }

    available_pairs = []
    for hsi_path in hsi_files:
        stem = os.path.splitext(os.path.basename(hsi_path))[0]
        rgb_path = rgb_lookup.get(stem)
        if rgb_path is not None:
            available_pairs.append((hsi_path, rgb_path))

    if num_samples > len(available_pairs):
        raise ValueError(
            f"Requested {num_samples} samples, but only "
            f"{len(available_pairs)} paired samples are available."
        )

    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(available_pairs)), num_samples)
    selected_remote_pairs = [available_pairs[i] for i in selected_indices]

    return selected_remote_pairs, selected_indices, len(available_pairs)


def load_random_arad1k_samples(
    root_dir="data",
    num_samples=50,
    seed=42,
    total_images=1000,
    cube_key="cube",
    download=True,
):
    """
    Return exactly ``num_samples`` random ARAD1K RGB-HSI pairs.

    Unlike the earlier implementation, this function samples the remote
    filenames first and downloads only the selected pairs. It therefore does
    not instantiate or scan a 1,000-image ARADDataset before taking a subset.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero.")

    selected_remote_pairs, selected_indices, available_count = _sample_remote_pairs(
        total_images=total_images,
        num_samples=num_samples,
        seed=seed,
    )

    local_pairs = []
    selected_samples = []

    for subset_index, ((hsi_remote, rgb_remote), dataset_index) in enumerate(
        zip(selected_remote_pairs, selected_indices)
    ):
        if download:
            hsi_local = hf_hub_download(
                repo_id="mhmdjouni/arad_hsdb",
                repo_type="dataset",
                filename=hsi_remote,
                local_dir=root_dir,
                local_dir_use_symlinks=False,
            )
            rgb_local = hf_hub_download(
                repo_id="mhmdjouni/arad_hsdb",
                repo_type="dataset",
                filename=rgb_remote,
                local_dir=root_dir,
                local_dir_use_symlinks=False,
            )
        else:
            hsi_local = os.path.join(root_dir, hsi_remote)
            rgb_local = os.path.join(root_dir, rgb_remote)

            if not os.path.isfile(hsi_local) or not os.path.isfile(rgb_local):
                raise FileNotFoundError(
                    "A selected ARAD1K pair is missing locally while "
                    "download=False:\n"
                    f"HSI: {hsi_local}\nRGB: {rgb_local}"
                )

        local_pairs.append((hsi_local, rgb_local))
        selected_samples.append(
            {
                "subset_index": subset_index,
                "dataset_index": dataset_index,
                "rgb_filename": os.path.basename(rgb_remote),
                "hsi_filename": os.path.basename(hsi_remote),
            }
        )

    dataset = RandomARAD1KDataset(
        pairs=local_pairs,
        selected_samples=selected_samples,
        cube_key=cube_key,
    )

    if len(dataset) != num_samples:
        raise RuntimeError(
            f"Expected exactly {num_samples} selected samples, "
            f"but constructed {len(dataset)}."
        )

    print(
        f"Selected and prepared exactly {len(dataset)} ARAD1K pairs "
        f"from {available_count} available pairs using seed {seed}."
    )

    return dataset, selected_samples
