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

"""State-dict transforms for loading ArtiFixer checkpoints into FlashDreams.

Two source layouts are supported:

  * **Vanilla Wan 2.1 1.3B** from ``Wan-AI/Wan2.1-T2V-1.3B`` -- the
    transform :func:`zero_pad_artifixer_keys` zero-pads the ArtiFixer-only
    keys so ``load_state_dict`` succeeds in strict mode. Zero-padding
    matches dreamfix's initialization in
    ``ArtifixerTransformerBlock.__init__`` L637-651 and produces a
    behaviorally-identical-to-Wan model (the ArtiFixer extension paths
    are zero-gated until trained weights are loaded).

  * **Merged ArtiFixer DMD safetensors** produced by
    ``dreamfix/scripts/merge_dcp_to_safetensors.py``. Built by
    :func:`artifixer_dmd_state_dict_transform`, which applies the
    HuggingFace diffusers -> WanDiTNetwork regex remap (same as
    ``fastvideo_causal_wan22.config.state_dict_transform``) plus the
    ArtiFixer ``attn2 -> cross_attn`` step that picks up
    ``add_k_proj`` / ``add_v_proj`` / ``norm_added_k``. The 270
    ArtiFixer-only keys map cleanly onto our new
    ``ArtifixerBlock`` / ``ArtifixerCrossAttention`` submodules without
    further per-key renames.
"""

from __future__ import annotations

import re
from typing import Callable

import torch
from torch import Tensor

from artifixer.network.dit import artifixer_embedding_dims

# HF diffusers ``WanTransformer3DModel`` -> flashdreams ``WanDiTNetwork``
# key remap. Verbatim copy of
# ``integrations/fastvideo_causal_wan22/.../config.py`` ``CHECKPOINT_KEY_MAPPING``
# (Wan 2.1 / 2.2 share this layout). The ArtiFixer-only keys --
# ``blocks.X.opacity_embedding.*``, ``blocks.X.camera_embedding.*``,
# ``blocks.X.attn2.add_k_proj.*``, ``blocks.X.attn2.add_v_proj.*``,
# ``blocks.X.attn2.norm_added_k.weight`` -- pass through the
# ``attn2 -> cross_attn`` substitution into the names
# :class:`ArtifixerBlock` / :class:`ArtifixerCrossAttention` register at
# ``__init__`` time. ``opacity_embedding`` / ``camera_embedding`` carry
# no ``attn2`` prefix so they fall through unchanged, which still
# matches the ArtifixerBlock attributes.
DIFFUSERS_TO_WAN_DIT_NETWORK_KEY_MAPPING: dict[str, str] = {
    r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": r"text_embedding.0.\1",
    r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": r"text_embedding.2.\1",
    r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": r"time_embedding.0.\1",
    r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": r"time_embedding.2.\1",
    r"^condition_embedder\.time_proj\.(.*)$": r"time_projection.1.\1",
    r"^scale_shift_table$": r"head.modulation",
    r"^proj_out\.(.*)$": r"head.head.\1",
    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.self_attn.q.\2",
    r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.self_attn.k.\2",
    r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.self_attn.v.\2",
    r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.self_attn.o.\2",
    r"^blocks\.(\d+)\.attn2\.to_q\.(.*)$": r"blocks.\1.cross_attn.q.\2",
    r"^blocks\.(\d+)\.attn2\.to_k\.(.*)$": r"blocks.\1.cross_attn.k.\2",
    r"^blocks\.(\d+)\.attn2\.to_v\.(.*)$": r"blocks.\1.cross_attn.v.\2",
    r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.cross_attn.o.\2",
    # ArtiFixer-only attn2 keys: ``add_k_proj`` / ``add_v_proj`` /
    # ``norm_added_k``. Same ``attn2 -> cross_attn`` substitution.
    r"^blocks\.(\d+)\.attn2\.add_k_proj\.(.*)$": r"blocks.\1.cross_attn.add_k_proj.\2",
    r"^blocks\.(\d+)\.attn2\.add_v_proj\.(.*)$": r"blocks.\1.cross_attn.add_v_proj.\2",
    r"^blocks\.(\d+)\.attn2\.norm_added_k\.(.*)$": r"blocks.\1.cross_attn.norm_added_k.\2",
    r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.self_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.self_attn.norm_k.\2",
    r"^blocks\.(\d+)\.attn2\.norm_q\.(.*)$": r"blocks.\1.cross_attn.norm_q.\2",
    r"^blocks\.(\d+)\.attn2\.norm_k\.(.*)$": r"blocks.\1.cross_attn.norm_k.\2",
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.norm3.\2",
    r"^blocks\.(\d+)\.scale_shift_table$": r"blocks.\1.modulation",
    r"^blocks\.(\d+)\.ffn\.fc_in\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.fc_out\.(.*)$": r"blocks.\1.ffn.2.\2",
    r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.0.\2",
    r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.2.\2",
}


def _remap_keys(
    state_dict: dict[str, Tensor], mapping: dict[str, str]
) -> dict[str, Tensor]:
    """Apply the first matching regex substitution to every key.

    Identical semantics to ``flashdreams.core.checkpoint.remap.remap_checkpoint_keys``;
    inlined here so this module has no dependency on flashdreams' internal
    helpers when used as a state_dict_transform callable.
    """
    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        for old_pattern, new_pattern in mapping.items():
            if re.match(old_pattern, k):
                new_k = re.sub(old_pattern, new_pattern, k)
                break
        out[new_k] = v
    return out


def artifixer_dmd_state_dict_transform(
    state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Remap a merged ArtiFixer DMD safetensors state_dict onto WanDiTNetwork.

    The merged safetensors produced by
    ``dreamfix/scripts/merge_dcp_to_safetensors.py`` carries the HF
    diffusers ``WanTransformer3DModel`` naming (e.g.
    ``blocks.X.attn1.to_q.weight``) plus 270 ArtiFixer-only keys with
    the ``attn2`` cross-attention prefix
    (``blocks.X.attn2.add_k_proj.weight``) and 60 keys without prefix
    (``blocks.X.opacity_embedding.weight``,
    ``blocks.X.camera_embedding.weight``).

    Apply :data:`DIFFUSERS_TO_WAN_DIT_NETWORK_KEY_MAPPING` to convert
    diffusers names to the flashdreams ``WanDiTNetwork`` /
    :class:`ArtifixerDiTNetwork` layout. The ArtiFixer-only keys land on
    :class:`ArtifixerBlock` / :class:`ArtifixerCrossAttention` attributes
    that are registered at ``__init__`` time, so ``load_state_dict``
    succeeds in strict mode with no remaining missing / unexpected keys.
    """
    return _remap_keys(state_dict, DIFFUSERS_TO_WAN_DIT_NETWORK_KEY_MAPPING)

def zero_pad_artifixer_keys(
    *,
    num_layers: int,
    dim: int,
    patch_size: tuple[int, int, int],
    dtype: torch.dtype = torch.bfloat16,
) -> Callable[[dict[str, Tensor]], dict[str, Tensor]]:
    """Build a state-dict transform that zero-fills ArtiFixer-only keys.

    Use when the source checkpoint is a vanilla Wan 2.1 base
    (e.g. ``Wan-AI/Wan2.1-T2V-1.3B``): it has 825 base keys per block but
    missing the ArtiFixer extensions, so :meth:`load_state_dict` in strict
    mode would error. We pre-add zero tensors so loading succeeds and the
    runtime behavior is unchanged versus vanilla Wan.

    Per block, the transform adds 9 keys (mirroring the 270 ArtiFixer-only
    keys identified by ``dreamfix/scripts/dump_artifixer_param_names.py``):

      * Phase 2.1 — opacity + camera MLPs (4 keys per block):
        ``opacity_embedding.{weight,bias}`` shape (dim, opacity_dim) / (dim,)
        ``camera_embedding.{weight,bias}``  shape (dim, camera_dim)  / (dim,)
      * Phase 2.2 — neighbor cross-attention (5 keys per block):
        ``cross_attn.add_k_proj.{weight,bias}``    shape (dim, dim) / (dim,)
        ``cross_attn.add_v_proj.{weight,bias}``    shape (dim, dim) / (dim,)
        ``cross_attn.norm_added_k.weight``         shape (dim,)

    Args:
        num_layers: Number of transformer blocks (1.3B Wan: 30).
        dim: Transformer hidden size (1.3B Wan: 1536).
        patch_size: Network ``patch_size`` (default ``(1, 2, 2)``).
        dtype: Dtype of the zero tensors (should match ``Wan21TransformerConfig.dtype``).
    """
    opacity_dim, camera_dim = artifixer_embedding_dims(patch_size)

    shapes: dict[str, tuple[int, ...]] = {}
    for layer in range(num_layers):
        prefix = f"blocks.{layer}."
        shapes[prefix + "opacity_embedding.weight"] = (dim, opacity_dim)
        shapes[prefix + "opacity_embedding.bias"] = (dim,)
        shapes[prefix + "camera_embedding.weight"] = (dim, camera_dim)
        shapes[prefix + "camera_embedding.bias"] = (dim,)
        shapes[prefix + "cross_attn.add_k_proj.weight"] = (dim, dim)
        shapes[prefix + "cross_attn.add_k_proj.bias"] = (dim,)
        shapes[prefix + "cross_attn.add_v_proj.weight"] = (dim, dim)
        shapes[prefix + "cross_attn.add_v_proj.bias"] = (dim,)
        shapes[prefix + "cross_attn.norm_added_k.weight"] = (dim,)

    def transform(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        out = dict(state_dict)
        for key, shape in shapes.items():
            if key not in out:
                out[key] = torch.zeros(shape, dtype=dtype)
        return out

    return transform


__all__ = [
    "DIFFUSERS_TO_WAN_DIT_NETWORK_KEY_MAPPING",
    "artifixer_dmd_state_dict_transform",
    "zero_pad_artifixer_keys",
]
