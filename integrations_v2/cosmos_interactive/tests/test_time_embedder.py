# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timestep embedding parity (DiT-style, matches Cosmos3 ``TimestepEmbedder``)."""

from __future__ import annotations

import math

import pytest
import torch
from cosmos_interactive._network import _TimeEmbedder, _timestep_embedding
from cosmos_interactive._spec import Cosmos3InteractiveSpec

pytestmark = pytest.mark.ci_cpu


def _reference_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Independent DiT reference (facebookresearch/DiT) for the sinusoidal part."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def test_sinusoidal_matches_dit_reference() -> None:
    t = torch.tensor([0.0, 0.5, 1.0, 123.4, 999.0])
    emb = _timestep_embedding(t, 256)
    ref = _reference_embedding(t, 256)
    assert emb.shape == (5, 256)
    assert (emb - ref).abs().max().item() < 1e-6


def test_t_zero_is_ones_then_zeros() -> None:
    emb = _timestep_embedding(torch.tensor([0.0]), 256)
    # cos(0)=1 for the first half, sin(0)=0 for the second half.
    assert torch.allclose(emb[0, :128], torch.ones(128))
    assert torch.allclose(emb[0, 128:], torch.zeros(128))


def test_time_embedder_forward_shape() -> None:
    spec = Cosmos3InteractiveSpec()
    emb = _TimeEmbedder(spec, torch.float32).eval()
    with torch.no_grad():
        out = emb(torch.tensor([0.0, 1.0, 2.0]))
    assert out.shape == (3, spec.llm.hidden_size)
    assert torch.isfinite(out).all()
