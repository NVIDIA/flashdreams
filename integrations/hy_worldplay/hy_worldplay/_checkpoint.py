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

"""HY-WorldPlay distilled-checkpoint state-dict transform (phase 2b.5b).

Drives the load of the distilled WAN-5B checkpoint into the
:class:`HyWorldPlayWanDiTNetwork` parameter tree built by 2b.3 + 2b.4.

The upstream distilled ``.pt`` is a top-level dict with two keys -
``generator`` (the FSDP-unwrapped weights from the live training rank)
and ``generator_ema`` (the same weights, but with the
``_fsdp_wrapped_module.`` prefix preserved). Upstream's
``wan/generate.py`` line 146-167 reads ``generator``, strips
``model.`` and ``_fsdp_wrapped_module.``, then ``load_state_dict``s
into a ``WanTransformer3DModel`` whose constructor has just been
extended via ``add_discrete_action_parameters()`` so the action-embed
MLP and the per-block ``to_out_prope`` linears exist as zero-init
slots ready for the load.

We mirror that load pipeline in flashdreams:

1. Pick ``state_dict["generator"]`` out of the raw ``.pt`` envelope.
2. Strip ``model.`` and ``_fsdp_wrapped_module.`` prefixes
   (training artefacts).
3. Apply the diffusers ``WanTransformer3DModel`` -> bare
   :class:`WanDiTNetwork` key remap from
   :data:`flashdreams.recipes.wan.config.wan22_ti2v_5b_dit_state_dict_transform`,
   extended with three HY-specific rewrite rules for
   ``action_embedder`` and ``to_out_prope`` so they land on our
   :attr:`HyWorldPlayWanDiTNetwork.action_embedding` and
   :attr:`HyWorldPlayPRoPESelfAttention.o_prope` parameters
   respectively.

The transform is structured so it accepts *both* the raw distilled
``.pt`` envelope (``generator`` subkey present) and a pre-stripped
state-dict (no envelope), which keeps it ergonomic for tests that
want to feed in synthetic dicts.
"""

from __future__ import annotations

import torch

from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.recipes.wan.config import wan22_ti2v_5b_dit_state_dict_transform

__all__ = [
    "hy_worldplay_distilled_state_dict_transform",
]


# Three HY-specific rewrite rules layered on top of the base
# Wan 2.2 TI2V 5B remap (which handles the standard
# ``WanTransformer3DModel`` <-> ``WanDiTNetwork`` mapping):
#
# * ``condition_embedder.action_embedder.linear_{1,2}`` ->
#   ``action_embedding.{0,2}`` (standard Wan-MLP indexing -- linear,
#   SiLU, linear; the SiLU at index 1 has no parameters so it's elided
#   from the remap).
# * ``blocks.{i}.attn1.to_out_prope.0`` ->
#   ``blocks.{i}.self_attn.o_prope`` (upstream's
#   ``to_out_prope`` is an ``nn.Sequential`` whose first element is
#   the linear; our ``HyWorldPlayPRoPESelfAttention.o_prope`` is the
#   linear directly, so we drop the ``.0.`` middle hop).
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
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap upstream's distilled WAN-5B checkpoint to HY ``WanDiTNetwork`` keys.

    Accepts either the raw envelope returned by ``torch.load`` on
    upstream's ``wan_distilled_model/model.pt`` (top-level
    ``generator`` / ``generator_ema`` subkeys) or a pre-stripped
    state-dict whose keys already start at the model root (e.g.
    ``model.blocks.0.attn1.to_q.weight`` or even
    ``blocks.0.attn1.to_q.weight``).

    Returns a flat ``dict[str, Tensor]`` keyed by the
    :class:`HyWorldPlayWanDiTNetwork` parameter names so
    :func:`torch.nn.Module.load_state_dict` can be called with
    ``strict=True``.
    """
    # 1. Unwrap the distilled-checkpoint envelope. ``generator`` (no
    # _fsdp_wrapped_module prefix) is what upstream uses; we mirror
    # that. ``generator_ema`` is the EMA copy and would be acceptable
    # too -- both contain the same logical weights -- but pinning to
    # ``generator`` keeps us bit-identical with upstream's load path.
    if "generator" in state_dict and "generator_ema" in state_dict:
        state_dict = state_dict["generator"]

    # 2. Strip training-time prefixes. ``model.`` is added by upstream's
    # outer ``WanTransformer3DModel``-wrapping training module;
    # ``_fsdp_wrapped_module.`` is the FSDP wrapper artefact. Order
    # matters: ``model.`` is always at the top; if present, we strip
    # it first; if ``_fsdp_wrapped_module.`` is left over, we strip it
    # next (FSDP wraps individual blocks, so it appears in the middle
    # of paths like ``blocks.0._fsdp_wrapped_module.attn1.to_q.weight``,
    # not just at the top).
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("model.").replace(
            "_fsdp_wrapped_module.", ""
        )
        stripped[new_key] = value

    # 3. Apply the base 5B diffusers -> WanDiTNetwork remap. This
    # covers everything except the HY-specific deltas (action +
    # PRoPE), which the downstream remap dict catches.
    base_remapped = wan22_ti2v_5b_dit_state_dict_transform(stripped)

    # 4. Apply the HY-specific rewrites. ``remap_checkpoint_keys``
    # is regex-rule-based and leaves keys it doesn't match alone, so
    # the base + HY rules compose cleanly.
    return remap_checkpoint_keys(base_remapped, _HY_WORLDPLAY_HY_KEY_REMAP)
