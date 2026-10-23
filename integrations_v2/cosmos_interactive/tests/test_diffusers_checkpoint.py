# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical attention parity for the GA Diffusers checkpoint mapping."""

import pytest
import torch
import torch.nn.functional as F
from cosmos_interactive._network import _MoTLayer
from cosmos_interactive._spec import LLMSpec
from cosmos_interactive.diffusers_checkpoint import remap_diffusers_key

pytestmark = pytest.mark.ci_cpu


def test_ga_attention_preserves_understanding_and_generation_streams() -> None:
    # Reference: Cosmos3PackedMoTAttnProcessor in the official implementation:
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos3.py
    spec = LLMSpec(
        hidden_size=8,
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        vocab_size=16,
        mrope_section=(1, 1, 0),
    )
    generator = torch.Generator().manual_seed(27)
    shapes = {
        "to_q": (8, 8),
        "to_k": (4, 8),
        "to_v": (4, 8),
        "to_out": (8, 8),
        "add_q_proj": (8, 8),
        "add_k_proj": (4, 8),
        "add_v_proj": (4, 8),
        "to_add_out": (8, 8),
        "norm_q": (4,),
        "norm_k": (4,),
        "norm_added_q": (4,),
        "norm_added_k": (4,),
    }
    weights = {
        name: torch.randn(shape, generator=generator) for name, shape in shapes.items()
    }
    layer = _MoTLayer(spec, torch.float32)
    remapped = {}
    for name, weight in weights.items():
        key = remap_diffusers_key(f"layers.0.self_attn.{name}.weight")
        assert key is not None
        remapped[key.removeprefix("layers.0.")] = weight
    layer.load_state_dict(remapped, strict=False)
    hidden = torch.randn((1, 5, 8), generator=generator)
    und, gen = hidden[:, :2], hidden[:, 2:]

    def project(tokens, q_name, k_name, v_name, q_norm, k_norm):
        q = F.linear(tokens, weights[q_name]).view(1, -1, 2, 4)
        k = F.linear(tokens, weights[k_name]).view(1, -1, 1, 4)
        v = F.linear(tokens, weights[v_name]).view(1, -1, 1, 4)
        q = F.rms_norm(q, (4,), weights[q_norm], spec.rms_norm_eps)
        k = F.rms_norm(k, (4,), weights[k_norm], spec.rms_norm_eps)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    uq, uk, uv = project(und, "to_q", "to_k", "to_v", "norm_q", "norm_k")
    gq, gk, gv = project(
        gen, "add_q_proj", "add_k_proj", "add_v_proj", "norm_added_q", "norm_added_k"
    )
    und_attention = F.scaled_dot_product_attention(
        uq, uk, uv, is_causal=True, enable_gqa=True
    )
    gen_attention = F.scaled_dot_product_attention(
        gq,
        torch.cat((uk, gk), dim=2),
        torch.cat((uv, gv), dim=2),
        enable_gqa=True,
    )
    expected = torch.cat(
        (
            F.linear(und_attention.transpose(1, 2).reshape(1, 2, 8), weights["to_out"]),
            F.linear(
                gen_attention.transpose(1, 2).reshape(1, 3, 8), weights["to_add_out"]
            ),
        ),
        dim=1,
    )
    is_gen = torch.tensor([False, False, True, True, True])
    mask = torch.ones((5, 5), dtype=torch.bool)
    mask[:2] = False
    mask[:2, :2] = torch.ones((2, 2), dtype=torch.bool).tril()
    attention = layer._joint_attn(
        hidden,
        is_gen,
        torch.ones((1, 5, 4)),
        torch.zeros((1, 5, 4)),
        mask,
    )
    actual = torch.where(
        is_gen[None, :, None],
        layer.gen.self_attn.o_proj(attention),
        layer.und.self_attn.o_proj(attention),
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
