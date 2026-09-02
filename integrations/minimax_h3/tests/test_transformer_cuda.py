# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity tests for the MiniMax H3 transformer."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Transformer3DModel,
)
from minimax_h3.transformer import MiniMaxH3TransformerConfig

pytestmark = pytest.mark.ci_gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_transformer_matches_official_h3_forward() -> None:
    """Prove state-dict and numerical compatibility on a tiny CUDA model."""
    architecture: dict[str, Any] = {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 16,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "ffn_dim": 32,
        "in_channels": 2,
        "audio_in_channels": 4,
        "patch_size": (1, 1, 1),
        "text_dim": 10,
        "freq_dim": 8,
        "time_embed_hidden_dim": 16,
        "time_embed_dim": 8,
        "rope_freq_dim": 2,
    }
    device = torch.device("cuda")
    torch.manual_seed(1)
    official: Any = MiniMaxH3Transformer3DModel(**architecture)
    official.to(device)
    native = MiniMaxH3TransformerConfig(
        checkpoint_path=None,
        device="cuda",
        execution_device="cuda",
        sequential_cpu_offload=False,
        dtype=torch.float32,
        attention_backend="math",
        **architecture,
    ).setup()
    native.load_state_dict(official.state_dict(), strict=True)
    inputs = {
        "hidden_states": torch.randn(1, 4, 2, device=device),
        "audio_hidden_states": torch.randn(1, 2, 4, device=device),
        "encoder_hidden_states": torch.randn(1, 3, 10, device=device),
        "timestep": torch.tensor([0.1, 0.5, 0.999], device=device),
        "timestep_indices": torch.tensor([1, 1, 1, 2, 0, 1, 1, 0, 2], device=device),
        "token_tags": torch.tensor([1, 1, 1, 0, 0, 2, 2, 0, 0], device=device),
        "position_ids": torch.randn(9, 3, device=device),
        "video_indices": torch.tensor([3, 4, 7, 8], device=device),
        "audio_indices": torch.tensor([5, 6], device=device),
        "text_indices": torch.tensor([0, 1, 2], device=device),
    }
    with (
        torch.no_grad(),
        torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
    ):
        expected = official(**inputs, return_dict=False)
        actual = native.forward_joint(**inputs)
    for native_output, official_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(native_output, official_output)
