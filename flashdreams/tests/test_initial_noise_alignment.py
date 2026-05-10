# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for explicit initial-noise alignment helpers."""

from __future__ import annotations

import pytest
import torch

from flashdreams.infra.diffusion.noise import (
    select_initial_noise_chunk,
    select_temporal_initial_noise_chunk,
    stack_initial_noise_chunks,
)


def test_select_initial_noise_chunk_from_temporal_rollout() -> None:
    latent_shape = (1, 3, 2, 2, 2)
    rollout = torch.arange(1 * 9 * 2 * 2 * 2).reshape(1, 9, 2, 2, 2)

    chunk = select_initial_noise_chunk(
        rollout,
        autoregressive_index=1,
        latent_shape=latent_shape,
    )

    torch.testing.assert_close(chunk, rollout[:, 3:6])


def test_select_initial_noise_chunk_from_chunk_stack() -> None:
    latent_shape = (1, 3, 2, 2, 2)
    rollout = torch.randn(4, *latent_shape)

    chunk = select_initial_noise_chunk(
        rollout,
        autoregressive_index=2,
        latent_shape=latent_shape,
    )

    torch.testing.assert_close(chunk, rollout[2])


def test_stack_initial_noise_chunks_uses_temporal_layout() -> None:
    chunks = [torch.full((1, 3, 1, 1, 1), i) for i in range(3)]

    rollout = stack_initial_noise_chunks(chunks)

    assert tuple(rollout.shape) == (1, 9, 1, 1, 1)
    torch.testing.assert_close(rollout[:, 3:6], chunks[1])


def test_select_temporal_initial_noise_chunk_for_patchified_models() -> None:
    chunk_shape = (1, 3, 2, 2, 2)
    rollout = torch.arange(1 * 9 * 2 * 2 * 2).reshape(1, 9, 2, 2, 2)

    chunk = select_temporal_initial_noise_chunk(
        rollout,
        autoregressive_index=2,
        chunk_shape=chunk_shape,
    )

    torch.testing.assert_close(chunk, rollout[:, 6:9])


def test_select_initial_noise_chunk_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="initial-noise rollout"):
        select_initial_noise_chunk(
            torch.randn(2, 3),
            autoregressive_index=0,
            latent_shape=(1, 3, 2, 2, 2),
        )
