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

"""Native and Diffusers AgiBot checkpoint keys for the MoT parameter tree."""

from __future__ import annotations

import re
from typing import TypeVar

_NET_PREFIX = "model.net."
_LM_PREFIX = "language_model."

# ``_moe_gen`` either suffixes a leaf module (``q_proj_moe_gen``) or a parent
# module (``mlp_moe_gen.gate_proj``). Strip both forms to recover the und leaf.
_MOE_GEN_RE = re.compile(r"_moe_gen(?=\.|$)")

_DIFFUSERS_ROOT_KEYS = {
    "embed_tokens.weight": "embed_tokens.weight",
    "lm_head.weight": "lm_head.weight",
    "norm.weight": "norm.weight",
    "norm_moe_gen.weight": "norm_moe_gen.weight",
    "action_modality_embed": "action_modality_embed",
    "action_proj_in.fc.weight": "action2llm.fc.weight",
    "action_proj_in.bias.weight": "action2llm.bias.weight",
    "action_proj_out.fc.weight": "llm2action.fc.weight",
    "action_proj_out.bias.weight": "llm2action.bias.weight",
    "proj_in.weight": "vae2llm.weight",
    "proj_in.bias": "vae2llm.bias",
    "proj_out.weight": "llm2vae.weight",
    "proj_out.bias": "llm2vae.bias",
    "time_embedder.linear_1.weight": "time_embedder.mlp.0.weight",
    "time_embedder.linear_1.bias": "time_embedder.mlp.0.bias",
    "time_embedder.linear_2.weight": "time_embedder.mlp.2.weight",
    "time_embedder.linear_2.bias": "time_embedder.mlp.2.bias",
}

_DIFFUSERS_LAYER_KEYS = {
    "input_layernorm.weight": "und.input_layernorm.weight",
    "input_layernorm_moe_gen.weight": "gen.input_layernorm.weight",
    "post_attention_layernorm.weight": "und.post_attention_layernorm.weight",
    "post_attention_layernorm_moe_gen.weight": "gen.post_attention_layernorm.weight",
    "mlp.gate_proj.weight": "und.mlp.gate_proj.weight",
    "mlp.up_proj.weight": "und.mlp.up_proj.weight",
    "mlp.down_proj.weight": "und.mlp.down_proj.weight",
    "mlp_moe_gen.gate_proj.weight": "gen.mlp.gate_proj.weight",
    "mlp_moe_gen.up_proj.weight": "gen.mlp.up_proj.weight",
    "mlp_moe_gen.down_proj.weight": "gen.mlp.down_proj.weight",
    "self_attn.to_q.weight": "und.self_attn.q_proj.weight",
    "self_attn.to_k.weight": "und.self_attn.k_proj.weight",
    "self_attn.to_v.weight": "und.self_attn.v_proj.weight",
    "self_attn.to_out.weight": "und.self_attn.o_proj.weight",
    "self_attn.norm_q.weight": "und.self_attn.q_norm.weight",
    "self_attn.norm_k.weight": "und.self_attn.k_norm.weight",
    "self_attn.add_q_proj.weight": "gen.self_attn.q_proj.weight",
    "self_attn.add_k_proj.weight": "gen.self_attn.k_proj.weight",
    "self_attn.add_v_proj.weight": "gen.self_attn.v_proj.weight",
    "self_attn.to_add_out.weight": "gen.self_attn.o_proj.weight",
    "self_attn.norm_added_q.weight": "gen.self_attn.q_norm.weight",
    "self_attn.norm_added_k.weight": "gen.self_attn.k_norm.weight",
}

T = TypeVar("T")


def remap_key(key: str) -> str | None:
    """Rename one native or Diffusers export key to its FlashDreams target name.

    Returns ``None`` for keys that are intentionally not part of the network
    weights (there are none expected today; ``None`` surfaces surprises).
    """
    if not key.startswith(_NET_PREFIX):
        if key in _DIFFUSERS_ROOT_KEYS:
            return _DIFFUSERS_ROOT_KEYS[key]
        layer = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
        if layer is not None:
            target = _DIFFUSERS_LAYER_KEYS.get(layer[2])
            if target is not None:
                return f"layers.{layer[1]}.{target}"
        return None
    rest = key[len(_NET_PREFIX) :]

    # VFM adapters + action-conditioning pathway live directly under ``model.net``.
    if rest.startswith(
        ("vae2llm.", "llm2vae.", "time_embedder.", "action2llm.", "llm2action.")
    ):
        return rest
    if rest == "action_modality_embed":  # bare nn.Parameter (camera checkpoint)
        return rest

    if not rest.startswith(_LM_PREFIX):
        return None
    rest = rest[len(_LM_PREFIX) :]  # after ``language_model.``

    # Vision (ViT) tower.
    if rest.startswith("visual."):
        return rest

    # Top-level LM head (sits at ``language_model.lm_head`` in the export).
    if rest.startswith("lm_head."):
        return rest

    # Transformer body: ``model.embed_tokens`` / ``model.norm`` / ``model.layers.*``.
    if rest.startswith("model."):
        body = rest[len("model.") :]
        if body.startswith("layers."):
            return _remap_layer(body)
        # embed_tokens.weight / norm.weight
        return body

    return None


def _remap_layer(body: str) -> str:
    """Map ``layers.{i}.<leaf>`` → ``layers.{i}.{und|gen}.<leaf>``."""
    m = re.match(r"^layers\.(\d+)\.(.+)$", body)
    if m is None:
        return body
    layer_idx, leaf = m.group(1), m.group(2)
    if _MOE_GEN_RE.search(leaf):
        pathway = "gen"
        leaf = _MOE_GEN_RE.sub("", leaf)
    else:
        pathway = "und"
    return f"layers.{layer_idx}.{pathway}.{leaf}"


def state_dict_transform(raw: dict[str, T]) -> dict[str, T]:
    """Rename every key in ``raw`` to the FlashDreams target name (tensors ride along).

    Raises ``KeyError`` if any source key has no mapping, so an unexpected key
    fails loudly rather than being silently dropped.
    """
    out: dict[str, T] = {}
    unmapped: list[str] = []
    for key, value in raw.items():
        target = remap_key(key)
        if target is None:
            unmapped.append(key)
            continue
        if target in out:
            raise KeyError(f"multiple checkpoint keys map to {target!r}")
        out[target] = value
    if unmapped:
        raise KeyError(
            f"{len(unmapped)} source keys have no target mapping, e.g. {unmapped[:5]}"
        )
    return out
