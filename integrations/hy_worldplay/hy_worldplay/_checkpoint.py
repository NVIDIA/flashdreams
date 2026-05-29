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

"""HY-WorldPlay distilled-checkpoint state-dict transform."""

from __future__ import annotations

from typing import Any

import torch
from wan22.config import wan22_ti2v_5b_dit_state_dict_transform

from flashdreams.core.checkpoint.remap import remap_checkpoint_keys

__all__ = [
    "hy_worldplay_distilled_state_dict_transform",
]


# Three HY-specific rewrite rules layered on top of the base
# Wan 2.2 TI2V 5B remap (which handles the standard
# ``WanTransformer3DModel`` <-> ``WanDiTNetwork`` mapping):
#
# * ``condition_embedder.action_embedder.linear_{1,2}`` ->
#   ``action_embedding.{0,2}`` -- standard Wan MLP indexing (linear,
#   SiLU, linear); the SiLU at index 1 has no parameters and is elided.
# * ``blocks.{i}.attn1.to_out_prope.0`` ->
#   ``blocks.{i}.self_attn.o_prope`` -- upstream's ``to_out_prope`` is
#   an ``nn.Sequential`` whose first element is the linear; our
#   :attr:`HyWorldPlayPRoPESelfAttention.o_prope` is the linear
#   directly, so we drop the ``.0.`` middle hop.
_HY_WORLDPLAY_HY_KEY_REMAP: dict[str, str] = {
    r"^condition_embedder\.action_embedder\.linear_1\.(.*)$": (
        r"action_embedding.0.\1"
    ),
    r"^condition_embedder\.action_embedder\.linear_2\.(.*)$": (
        r"action_embedding.2.\1"
    ),
    r"^blocks\.(\d+)\.attn1\.to_out_prope\.0\.(.*)$": (
        r"blocks.\1.self_attn.o_prope.\2"
    ),
}


def hy_worldplay_distilled_state_dict_transform(
    state_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Remap the distilled WAN-5B checkpoint to :class:`HyWorldPlayWanDiTNetwork` keys.

    Accepts either the raw envelope returned by ``torch.load`` on
    upstream's ``wan_distilled_model/model.pt`` (top-level
    ``generator`` / ``generator_ema`` subkeys) or a pre-stripped
    state-dict whose keys already start at the model root (e.g.
    ``model.blocks.0.attn1.to_q.weight`` or even
    ``blocks.0.attn1.to_q.weight``).

    Returns:
        Flat ``dict[str, Tensor]`` keyed by
        :class:`HyWorldPlayWanDiTNetwork` parameter names; safe to load
        with :meth:`torch.nn.Module.load_state_dict` under
        ``strict=True``.
    """
    # 1. Unwrap the distilled-checkpoint envelope. Pinning to
    # ``generator`` (not the EMA copy) matches upstream's load path
    # bit-for-bit.
    if "generator" in state_dict and "generator_ema" in state_dict:
        state_dict = state_dict["generator"]

    # 2. Strip training-time prefixes. ``model.`` is the outer training
    # module's wrapper; ``_fsdp_wrapped_module.`` is the FSDP artefact
    # and can appear anywhere along a key (FSDP wraps individual
    # blocks, e.g. ``blocks.0._fsdp_wrapped_module.attn1.to_q.weight``),
    # so it gets a global string replace rather than a prefix strip.
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("model.").replace("_fsdp_wrapped_module.", "")
        stripped[new_key] = value

    # 3. Apply the base 5B diffusers -> WanDiTNetwork remap. Covers
    # everything except the HY-specific action / PRoPE deltas handled
    # in step 4.
    base_remapped = wan22_ti2v_5b_dit_state_dict_transform(stripped)

    # 4. Layer the HY-specific rewrites on top. ``remap_checkpoint_keys``
    # is regex-rule-based and leaves non-matching keys alone, so the
    # base + HY rules compose cleanly.
    return remap_checkpoint_keys(base_remapped, _HY_WORLDPLAY_HY_KEY_REMAP)
