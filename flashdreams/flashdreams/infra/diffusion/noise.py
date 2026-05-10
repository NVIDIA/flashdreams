# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for explicit initial-noise reproduction runs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


def load_initial_noise_rollout(path: Path) -> Tensor:
    """Load a CPU noise rollout saved by FlashDreams or an external runner."""
    noise = torch.load(path, map_location="cpu")
    if not isinstance(noise, Tensor):
        raise TypeError(f"initial-noise file must contain a Tensor, got {type(noise)}")
    return noise


def select_initial_noise_chunk(
    rollout: Tensor,
    *,
    autoregressive_index: int,
    latent_shape: tuple[int, ...],
) -> Tensor:
    """Select one AR chunk from already-patchified rollout noise.

    Accepted layouts:
    - ``[num_chunks, *latent_shape]``
    - ``[B, num_chunks * T, C, H, W]`` when the model latent shape itself is
      unpatchified ``[B, T, C, H, W]``
    """
    if rollout.ndim == len(latent_shape) + 1:
        if tuple(rollout.shape[1:]) != latent_shape:
            raise ValueError(
                "chunk-stacked initial noise has incompatible chunk shape: "
                f"got {tuple(rollout.shape[1:])}, expected {latent_shape}"
            )
        if autoregressive_index >= rollout.shape[0]:
            raise IndexError(
                f"initial-noise rollout has {rollout.shape[0]} chunks, "
                f"requested chunk {autoregressive_index}"
            )
        return rollout[autoregressive_index]

    if rollout.ndim == len(latent_shape):
        if len(latent_shape) != 5:
            raise ValueError(
                "temporal-stacked initial noise currently expects latent_shape "
                f"[B, T, C, H, W], got {latent_shape}"
            )
        b, chunk_t, c, h, w = latent_shape
        if (
            rollout.shape[0] != b
            or rollout.shape[2] != c
            or rollout.shape[3] != h
            or rollout.shape[4] != w
        ):
            raise ValueError(
                "temporal-stacked initial noise has incompatible shape: "
                f"got {tuple(rollout.shape)}, expected [B, total_T, C, H, W] "
                f"with B/C/H/W from {latent_shape}"
            )
        start = autoregressive_index * chunk_t
        end = start + chunk_t
        if end > rollout.shape[1]:
            raise IndexError(
                f"initial-noise rollout has T={rollout.shape[1]}, "
                f"requested temporal slice [{start}:{end}]"
            )
        return rollout[:, start:end]

    raise ValueError(
        "initial-noise rollout must be either chunk-stacked or temporal-stacked: "
        f"got shape {tuple(rollout.shape)}, latent_shape {latent_shape}"
    )


def select_temporal_initial_noise_chunk(
    rollout: Tensor,
    *,
    autoregressive_index: int,
    chunk_shape: tuple[int, ...],
) -> Tensor:
    """Select one unpatchified chunk from ``[B, total_T, C, H, W]`` noise."""
    if len(chunk_shape) != 5:
        raise ValueError(f"chunk_shape must be [B, T, C, H, W], got {chunk_shape}")
    if rollout.ndim != 5:
        raise ValueError(
            "temporal initial noise must have shape [B, total_T, C, H, W], "
            f"got {tuple(rollout.shape)}"
        )

    b, chunk_t, c, h, w = chunk_shape
    if (
        rollout.shape[0] != b
        or rollout.shape[2] != c
        or rollout.shape[3] != h
        or rollout.shape[4] != w
    ):
        raise ValueError(
            "temporal initial noise has incompatible shape: "
            f"got {tuple(rollout.shape)}, expected [B, total_T, C, H, W] "
            f"with B/C/H/W from {chunk_shape}"
        )
    start = autoregressive_index * chunk_t
    end = start + chunk_t
    if end > rollout.shape[1]:
        raise IndexError(
            f"temporal initial-noise rollout has T={rollout.shape[1]}, "
            f"requested slice [{start}:{end}]"
        )
    return rollout[:, start:end]


def stack_initial_noise_chunks(chunks: list[Tensor]) -> Tensor:
    """Save chunks in official-friendly ``[B, total_T, C, H, W]`` layout."""
    if not chunks:
        raise ValueError("cannot stack an empty initial-noise chunk list")
    return torch.cat(chunks, dim=1)
