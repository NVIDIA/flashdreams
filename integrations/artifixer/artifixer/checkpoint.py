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

  * **Vanilla Wan 2.1 1.3B** from ``Wan-AI/Wan2.1-T2V-1.3B`` (used in
    Phase 1 / 2.1 while only the base architecture is implemented). The
    transform zero-pads the ArtiFixer-only keys so ``load_state_dict``
    succeeds in strict mode. Zero-padding matches the dreamfix
    initialization in ``ArtifixerTransformerBlock.__init__`` L637-651.

  * **Merged ArtiFixer DMD safetensors** produced by
    ``dreamfix/scripts/merge_dcp_to_safetensors.py``. This still uses the
    HuggingFace diffusers naming (e.g. ``blocks.X.attn1.to_q.weight``);
    Phase 5 will add the diffusers -> ``WanDiTNetwork`` regex remap on
    top of the zero-pad pass.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from artifixer.network.dit import artifixer_embedding_dims

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
    "zero_pad_artifixer_keys",
]
