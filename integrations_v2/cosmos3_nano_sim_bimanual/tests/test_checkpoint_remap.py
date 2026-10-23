# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict checkpoint key compatibility between native and Diffusers exports."""

import pytest
import torch
from cosmos3_nano_sim_bimanual.checkpoint_remap import (
    remap_key,
    state_dict_transform,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("diffusers", "native", "target"),
    [
        ("proj_in.weight", "vae2llm.weight", "vae2llm.weight"),
        ("proj_out.bias", "llm2vae.bias", "llm2vae.bias"),
        ("action_proj_in.fc.weight", "action2llm.fc.weight", "action2llm.fc.weight"),
        (
            "action_proj_out.bias.weight",
            "llm2action.bias.weight",
            "llm2action.bias.weight",
        ),
        ("action_modality_embed", "action_modality_embed", "action_modality_embed"),
        (
            "time_embedder.linear_2.bias",
            "time_embedder.mlp.2.bias",
            "time_embedder.mlp.2.bias",
        ),
        (
            "embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "embed_tokens.weight",
        ),
        ("lm_head.weight", "language_model.lm_head.weight", "lm_head.weight"),
        (
            "norm_moe_gen.weight",
            "language_model.model.norm_moe_gen.weight",
            "norm_moe_gen.weight",
        ),
        (
            "layers.3.self_attn.to_q.weight",
            "language_model.model.layers.3.self_attn.q_proj.weight",
            "layers.3.und.self_attn.q_proj.weight",
        ),
        (
            "layers.3.self_attn.add_q_proj.weight",
            "language_model.model.layers.3.self_attn.q_proj_moe_gen.weight",
            "layers.3.gen.self_attn.q_proj.weight",
        ),
        (
            "layers.3.self_attn.norm_added_k.weight",
            "language_model.model.layers.3.self_attn.k_norm_moe_gen.weight",
            "layers.3.gen.self_attn.k_norm.weight",
        ),
        (
            "layers.3.input_layernorm_moe_gen.weight",
            "language_model.model.layers.3.input_layernorm_moe_gen.weight",
            "layers.3.gen.input_layernorm.weight",
        ),
        (
            "layers.3.post_attention_layernorm.weight",
            "language_model.model.layers.3.post_attention_layernorm.weight",
            "layers.3.und.post_attention_layernorm.weight",
        ),
        (
            "layers.3.mlp_moe_gen.gate_proj.weight",
            "language_model.model.layers.3.mlp_moe_gen.gate_proj.weight",
            "layers.3.gen.mlp.gate_proj.weight",
        ),
    ],
)
def test_native_and_diffusers_keys_preserve_tensor_values(
    diffusers: str, native: str, target: str
) -> None:
    value = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    original = value.clone()
    for source in (diffusers, f"model.net.{native}"):
        raw = {source: value}
        remapped = state_dict_transform(raw)
        assert list(remapped) == [target]
        assert remapped[target] is value
        assert list(raw) == [source]
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "key",
    [
        "unexpected.weight",
        "proj_in.unexpected",
        "action_proj_in.weight",
        "time_embedder.linear_3.weight",
        "layers.bad.self_attn.to_q.weight",
        "layers.0.self_attn.to_q.bias",
        "layers.0.self_attn.unknown.weight",
        "layers.0.mlp.unknown.weight",
        "transformer.embed_tokens.weight",
    ],
)
def test_unknown_diffusers_keys_are_rejected(key: str) -> None:
    assert remap_key(key) is None
    with pytest.raises(KeyError, match="no target mapping"):
        state_dict_transform({key: object()})


def test_native_and_diffusers_collisions_are_rejected() -> None:
    with pytest.raises(KeyError, match="multiple checkpoint keys"):
        state_dict_transform(
            {
                "proj_in.weight": object(),
                "model.net.vae2llm.weight": object(),
            }
        )
