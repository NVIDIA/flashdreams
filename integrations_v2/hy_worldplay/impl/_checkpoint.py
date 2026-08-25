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
    "HY_WORLDPLAY_DISTILLED_CKPT_PATH",
    "hy_worldplay_distilled_state_dict_transform",
]

HY_WORLDPLAY_DISTILLED_CKPT_PATH = (
    "https://huggingface.co/tencent/HY-WorldPlay/resolve/main/"
    "wan_distilled_model/model.pt"
)
"""Default distilled HY-WorldPlay WAN-5B checkpoint (HF ``tencent/HY-WorldPlay``).

Loaded via :func:`hy_worldplay_distilled_state_dict_transform`. The repo is
gated, so set ``HF_TOKEN`` to download. Override with ``--ckpt-path`` to point
at a local ``model.pt``.
"""


# HY-specific rewrites layered on the base Wan 2.2 TI2V-5B remap:
# * ``action_embedder.linear_{1,2}`` -> ``action_embedding.{0,2}``
#   (Wan MLP indexing; the parameterless SiLU at index 1 is elided).
# * ``attn1.to_out_prope.0`` -> ``self_attn.o_prope`` -- upstream's
#   ``to_out_prope`` is an ``nn.Sequential``; ``o_prope`` is the bare
#   linear, so drop the ``.0.`` hop.
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

    Accepts either the raw ``torch.load`` envelope (top-level
    ``generator`` / ``generator_ema`` subkeys) or a state-dict whose
    keys already start at the model root.

    Returns:
        Flat ``dict[str, Tensor]`` keyed by
        :class:`HyWorldPlayWanDiTNetwork` parameter names; loadable
        under ``strict=True``.
    """
    # Unwrap the envelope; pin to ``generator`` (not the EMA copy).
    if "generator" in state_dict and "generator_ema" in state_dict:
        state_dict = state_dict["generator"]

    # Strip training-time prefixes. ``_fsdp_wrapped_module.`` can appear
    # mid-key (FSDP wraps individual blocks), so replace it globally.
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("model.").replace("_fsdp_wrapped_module.", "")
        stripped[new_key] = value

    # Apply the base 5B diffusers -> WanDiTNetwork remap, then the
    # HY-specific rules; regex-rule remapping leaves non-matching keys
    # alone, so the two compose cleanly.
    base_remapped = wan22_ti2v_5b_dit_state_dict_transform(stripped)
    return remap_checkpoint_keys(base_remapped, _HY_WORLDPLAY_HY_KEY_REMAP)
