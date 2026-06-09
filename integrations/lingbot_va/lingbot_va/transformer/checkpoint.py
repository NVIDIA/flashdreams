"""Checkpoint weight remapping from upstream LingBot-VA to native network."""

from __future__ import annotations

import re
from collections import OrderedDict

from torch import Tensor

_BLOCK_RULES: list[tuple[str, str]] = [
    (r"^blocks\.(\d+)\.attn1\.to_q\.(.+)$", r"blocks.\1.self_attn.q.\2"),
    (r"^blocks\.(\d+)\.attn1\.to_k\.(.+)$", r"blocks.\1.self_attn.k.\2"),
    (r"^blocks\.(\d+)\.attn1\.to_v\.(.+)$", r"blocks.\1.self_attn.v.\2"),
    (r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.+)$", r"blocks.\1.self_attn.o.\2"),
    (r"^blocks\.(\d+)\.attn1\.norm_q\.(.+)$", r"blocks.\1.self_attn.norm_q.\2"),
    (r"^blocks\.(\d+)\.attn1\.norm_k\.(.+)$", r"blocks.\1.self_attn.norm_k.\2"),
    (r"^blocks\.(\d+)\.attn2\.to_q\.(.+)$", r"blocks.\1.cross_attn.q.\2"),
    (r"^blocks\.(\d+)\.attn2\.to_k\.(.+)$", r"blocks.\1.cross_attn.k.\2"),
    (r"^blocks\.(\d+)\.attn2\.to_v\.(.+)$", r"blocks.\1.cross_attn.v.\2"),
    (r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.+)$", r"blocks.\1.cross_attn.o.\2"),
    (r"^blocks\.(\d+)\.attn2\.norm_q\.(.+)$", r"blocks.\1.cross_attn.norm_q.\2"),
    (r"^blocks\.(\d+)\.attn2\.norm_k\.(.+)$", r"blocks.\1.cross_attn.norm_k.\2"),
    (r"^blocks\.(\d+)\.norm2\.(.+)$", r"blocks.\1.norm3.\2"),
    (r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.+)$", r"blocks.\1.ffn.0.\2"),
    (r"^blocks\.(\d+)\.ffn\.net\.2\.(.+)$", r"blocks.\1.ffn.2.\2"),
    (r"^blocks\.(\d+)\.scale_shift_table$", r"blocks.\1.modulation"),
]

_TOPLEVEL_RULES: list[tuple[str, str]] = [
    (r"^patch_embedding_mlp\.(.+)$", r"patch_embedding.\1"),
    (r"^condition_embedder\.time_embedder\.linear_1\.(.+)$", r"time_embedding.0.\1"),
    (r"^condition_embedder\.time_embedder\.linear_2\.(.+)$", r"time_embedding.2.\1"),
    (r"^condition_embedder\.time_proj\.(.+)$", r"time_projection.1.\1"),
    (r"^condition_embedder\.text_embedder\.linear_1\.(.+)$", r"text_embedding.0.\1"),
    (r"^condition_embedder\.text_embedder\.linear_2\.(.+)$", r"text_embedding.2.\1"),
    (r"^action_embedder\.(.+)$", r"action_embedder.\1"),
    (r"^action_proj_out\.(.+)$", r"action_head.\1"),
    (r"^condition_embedder_action\.time_embedder\.linear_1\.(.+)$", r"action_time_embedding.0.\1"),
    (r"^condition_embedder_action\.time_embedder\.linear_2\.(.+)$", r"action_time_embedding.2.\1"),
    (r"^condition_embedder_action\.time_proj\.(.+)$", r"action_time_projection.1.\1"),
    (r"^condition_embedder_action\.text_embedder\.linear_1\.(.+)$", r"action_text_embedding.0.\1"),
    (r"^condition_embedder_action\.text_embedder\.linear_2\.(.+)$", r"action_text_embedding.2.\1"),
    (r"^proj_out\.(.+)$", r"head.head.\1"),
    (r"^scale_shift_table$", r"head.modulation"),
]

_DROP_PREFIXES = (
    "patch_embedding.",
    "norm_out.",
    "rope.",
    "condition_embedder.timesteps_proj.",
    "condition_embedder_action.timesteps_proj.",
)

_ALL_RULES = _BLOCK_RULES + _TOPLEVEL_RULES


def state_dict_transform(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Remap upstream LingBot-VA checkpoint keys to native WanVADiTNetwork keys."""
    remapped: dict[str, Tensor] = OrderedDict()

    for key, tensor in state_dict.items():
        if any(key.startswith(p) for p in _DROP_PREFIXES):
            continue

        new_key: str | None = None
        for pattern, replacement in _ALL_RULES:
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                break

        if new_key is None:
            raise ValueError(
                f"Unmapped checkpoint key: {key!r}. "
                "Update the remapping rules in checkpoint.py."
            )
        remapped[new_key] = tensor

    return remapped
