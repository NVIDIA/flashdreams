# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoT layer forward: per-token und/gen routing around the joint attention.

Two GPU-free checks:
  * **und parity** — an all-understanding sequence (full mask) routes through the
    und pathway and matches ``transformers.Qwen3DecoderLayer`` exactly.
  * **und-invariance** — an understanding token's output is unchanged when
    generation tokens are appended (proves the visibility mask really blocks
    und→gen and that und routing ignores gen tokens).
"""

from __future__ import annotations

import pytest
import torch
from cosmos_interactive._attention import build_mot_visibility_mask
from cosmos_interactive._network import _MoTLayer
from cosmos_interactive._spec import LLMSpec

pytestmark = pytest.mark.ci_cpu

_LLM = LLMSpec(
    hidden_size=64,
    num_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    vocab_size=100,
    mrope_section=(4, 2, 2),
    rope_theta=1_000_000.0,
)


def test_mot_layer_all_und_matches_hf_qwen3() -> None:
    pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer,
        Qwen3RotaryEmbedding,
    )

    torch.manual_seed(0)
    layer = _MoTLayer(_LLM, torch.float32).eval()

    cfg = Qwen3Config.from_dict(
        dict(
            hidden_size=_LLM.hidden_size,
            num_hidden_layers=1,
            num_attention_heads=_LLM.num_attention_heads,
            num_key_value_heads=_LLM.num_key_value_heads,
            head_dim=_LLM.head_dim,
            intermediate_size=_LLM.intermediate_size,
            vocab_size=_LLM.vocab_size,
            rms_norm_eps=_LLM.rms_norm_eps,
            rope_theta=_LLM.rope_theta,
            attention_bias=False,
            hidden_act="silu",
            max_position_embeddings=128,
            attn_implementation="eager",
        )
    )
    hf = Qwen3DecoderLayer(cfg, layer_idx=0).eval()
    hf.load_state_dict(layer.und.state_dict())  # HF mirrors the und pathway

    rotary = Qwen3RotaryEmbedding(cfg)
    b, s = 2, 5
    x = torch.randn(b, s, _LLM.hidden_size)
    pos = torch.arange(s)[None].expand(b, s)
    cos, sin = rotary(x, pos)

    # All-understanding, full attention (order constant -> every und key visible).
    is_gen = torch.zeros(s, dtype=torch.bool)
    order = torch.zeros(s, dtype=torch.long)
    frame = torch.zeros(s, dtype=torch.long)
    mask = build_mot_visibility_mask(is_gen, order, frame)

    with torch.no_grad():
        mine = layer(x, is_gen, cos, sin, mask)
        ref = hf(x, attention_mask=None, position_embeddings=(cos, sin))
    if isinstance(ref, tuple):
        ref = ref[0]
    assert (mine - ref).abs().max().item() < 1e-4


def test_mot_layer_und_output_invariant_to_gen_tokens() -> None:
    torch.manual_seed(0)
    layer = _MoTLayer(_LLM, torch.float32).eval()
    d = _LLM.head_dim

    # Layout: 3 und (text) tokens, then 2 gen tokens (one frame).
    n_und = 3
    is_gen = torch.tensor([False, False, False, True, True])
    order = torch.tensor([0, 1, 2, 3, 3])
    frame = torch.tensor([-1, -1, -1, 0, 0])
    mask = build_mot_visibility_mask(is_gen, order, frame)

    s_full = is_gen.numel()
    x_full = torch.randn(1, s_full, _LLM.hidden_size)
    cos = torch.randn(1, s_full, d)
    sin = torch.randn(1, s_full, d)

    # und-only run: first n_und tokens, causal among und.
    is_gen_u = is_gen[:n_und]
    order_u = order[:n_und]
    frame_u = frame[:n_und]
    mask_u = build_mot_visibility_mask(is_gen_u, order_u, frame_u)

    with torch.no_grad():
        out_full = layer(x_full, is_gen, cos, sin, mask)
        out_und = layer(
            x_full[:, :n_und], is_gen_u, cos[:, :n_und], sin[:, :n_und], mask_u
        )

    max_abs = (out_full[:, :n_und] - out_und).abs().max().item()
    assert max_abs < 1e-5, f"und output leaked from gen tokens: {max_abs}"
