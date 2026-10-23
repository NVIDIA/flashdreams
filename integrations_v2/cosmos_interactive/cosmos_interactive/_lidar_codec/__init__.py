# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load local LiDAR inference weights without the training repository."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from .network import TransformerVAE


def network_config(version: str) -> dict:
    """Select the trained architecture explicitly, never from a filename."""
    if version == "v1p1":
        return dict(resolution=(128, 1800), depths=(3, 3, 3), num_heads=(4, 4, 8))
    if version == "v1p2":
        return dict(resolution=(128, 1808), depths=(3, 3, 3, 3), num_heads=(4, 4, 8, 8))
    raise ValueError("LiDAR tokenizer version must be v1p1 or v1p2")


@dataclass(frozen=True)
class RangeProjection:
    """Separate calibrated rays from the circularly padded network canvas."""

    model_width: int
    semantic_height: int = 128
    semantic_width: int = 1800
    min_range_m: float = 5.0
    max_range_m: float = 100.0
    validity_threshold: float = 0.5


def load_checkpoint(model: TransformerVAE, path: str) -> tuple[Tensor, Tensor]:
    """Load trusted local weights and validate their latent statistics.

    Training .pt files contain OmegaConf metadata and require full unpickling;
    only load trusted checkpoints. Safetensors exports are also supported.
    No registry lookup, download, credential access, or architecture guessing
    occurs here. Missing, duplicate, unexpected, or mismatched network weights
    fail before the model is changed. Recognized training EMA copies are ignored;
    inference always selects the explicit network weights.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Local LiDAR checkpoint not found: {resolved}")
    if resolved.suffix == ".safetensors":
        from safetensors.torch import load_file

        payload = load_file(str(resolved), device="cpu")
    else:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("LiDAR checkpoint must be a mapping")
    raw_state = payload.get("model", payload)
    if not isinstance(raw_state, Mapping) or any(
        not isinstance(key, str) for key in raw_state
    ):
        raise TypeError(
            "LiDAR checkpoint model state must be a mapping with string keys"
        )
    state = cast(Mapping[str, Any], raw_state)

    def statistic(name: str, *, required: bool = True) -> Tensor | None:
        keys = [key for key in (name, f"model.{name}") if key in state]
        if len(keys) > 1:
            raise ValueError(f"Duplicate LiDAR statistic: {name}")
        if not keys:
            if required:
                raise ValueError(f"LiDAR checkpoint is missing {name}")
            return None
        return torch.as_tensor(state[keys[0]])

    channels = model.z_dim
    mean, std = statistic("latent_mean"), statistic("latent_std")
    assert mean is not None and std is not None
    mean, std = mean.float(), std.float()
    if mean.numel() != channels or std.numel() != channels:
        raise ValueError(f"LiDAR latent statistics must have {channels} values")
    if (
        not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError(
            "LiDAR latent statistics must be finite with positive standard deviations"
        )
    initialized = statistic("_latent_stats_initialized", required=False)
    if initialized is not None and (initialized.numel() != 1 or not initialized.item()):
        raise ValueError("LiDAR checkpoint latent statistics are not initialized")
    lo, hi = (
        statistic("min_range", required=False),
        statistic("max_range", required=False),
    )
    if (lo is None) != (hi is None):
        raise ValueError("LiDAR checkpoint must contain both min_range and max_range")
    # The previous wrapper supplied these explicit limits for legacy weights
    # lacking range metadata. Preserve that contract; reject contrary metadata.
    if (
        lo is not None
        and hi is not None
        and (
            lo.numel() != 1 or hi.numel() != 1 or lo.item() != 5.0 or hi.item() != 100.0
        )
    ):
        raise ValueError(
            "LiDAR checkpoint range must match the supported [5, 100] metre span"
        )

    expected = model.state_dict()
    weights = {}
    unexpected = []
    metadata = {
        "latent_mean",
        "latent_std",
        "min_range",
        "max_range",
        "_latent_stats_initialized",
    }
    # Training checkpoints store a second, flattened copy of network parameters.
    # Ignore only recognized EMA names; never substitute them for missing weights.
    metadata.update(
        "ema.network-" + name.replace(".", "-") for name, _ in model.named_parameters()
    )
    prefixes = ("model.network.", "network.", "model.vae.", "vae.")
    for original, value in state.items():
        if original.removeprefix("model.") in metadata or not isinstance(value, Tensor):
            continue
        key = original
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        if key not in expected:
            unexpected.append(original)
        elif key in weights:
            raise ValueError(f"Duplicate LiDAR network weight: {key}")
        else:
            weights[key] = value
    missing = sorted(set(expected) - set(weights))
    mismatched = {
        key: (tuple(value.shape), tuple(expected[key].shape))
        for key, value in weights.items()
        if value.shape != expected[key].shape
    }
    if missing or mismatched or unexpected:
        raise RuntimeError(
            "Incompatible LiDAR tokenizer checkpoint: "
            f"missing={missing[:20]}, mismatched={mismatched}, unexpected={unexpected[:20]}"
        )
    model.load_state_dict(weights, strict=True)
    return mean.float().reshape(1, channels, 1, 1, 1), std.float().reshape(
        1, channels, 1, 1, 1
    )


class LidarCodec:
    """Own the native network and checkpoint-specific latent affine."""

    def __init__(self, checkpoint: str, *, version: str, device: str):
        config = network_config(version)
        self.device, self.dtype = torch.device(device), torch.float32
        self.range_projection = RangeProjection(model_width=config["resolution"][1])
        self.min_range = self.range_projection.min_range_m
        self.max_range = self.range_projection.max_range_m
        self.validity_threshold = self.range_projection.validity_threshold
        self.model = TransformerVAE(**config)
        mean, std = load_checkpoint(self.model, checkpoint)
        self.model.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        self.latent_mean = mean.to(device=self.device, dtype=self.dtype)
        self.latent_std = std.to(device=self.device, dtype=self.dtype)
