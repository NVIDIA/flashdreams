# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 Diffusers checkpoint names for the native MoT parameter tree."""

import re

_PROJECTIONS = {
    "proj_in": "vae2llm",
    "proj_out": "llm2vae",
    "lidar_proj_in": "lidar2llm",
    "lidar_proj_out": "llm2lidar",
    "action_proj_in": "action2llm",
    "action_proj_out": "llm2action",
    "time_embedder.linear_1": "time_embedder.mlp.0",
    "time_embedder.linear_2": "time_embedder.mlp.2",
}

_ATTENTION = {
    "to_q": ("und", "q_proj"),
    "to_k": ("und", "k_proj"),
    "to_v": ("und", "v_proj"),
    "to_out": ("und", "o_proj"),
    "norm_q": ("und", "q_norm"),
    "norm_k": ("und", "k_norm"),
    "add_q_proj": ("gen", "q_proj"),
    "add_k_proj": ("gen", "k_proj"),
    "add_v_proj": ("gen", "v_proj"),
    "to_add_out": ("gen", "o_proj"),
    "norm_added_q": ("gen", "q_norm"),
    "norm_added_k": ("gen", "k_norm"),
}


def remap_diffusers_key(key: str) -> str | None:
    """Map one GA Diffusers weight name, rejecting unknown parameter names."""
    for source, target in _PROJECTIONS.items():
        if key.startswith(source + "."):
            return target + key[len(source) :]
    if key == "action_modality_embed" or key.startswith(
        ("embed_tokens.", "lm_head.", "norm.", "norm_moe_gen.", "rig_view_embed.")
    ):
        return key
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
    if match is None:
        return None
    layer, leaf = match.groups()
    if leaf.startswith("self_attn."):
        _, name, parameter = leaf.split(".", 2)
        mapping = _ATTENTION.get(name)
        if mapping is None:
            return None
        pathway, target = mapping
        return f"layers.{layer}.{pathway}.self_attn.{target}.{parameter}"
    if leaf.split(".", 1)[0] not in {
        "input_layernorm",
        "input_layernorm_moe_gen",
        "post_attention_layernorm",
        "post_attention_layernorm_moe_gen",
        "mlp",
        "mlp_moe_gen",
    }:
        return None
    pathway = "gen" if "_moe_gen." in leaf else "und"
    return f"layers.{layer}.{pathway}.{leaf.replace('_moe_gen.', '.')}"
