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

"""DiT block with per-block Plücker camera-control cross-attention."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
)


class CamCtrlBlock(Block):
    """Wan 2.1 transformer block + per-block camera-control branch."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        cp_method: Literal["ring", "ulysses"] = "ring",
    ) -> None:
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            cp_method=cp_method,
        )
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        plucker_embedding: Tensor,
    ) -> Tensor:
        """Run one transformer block update.

        Args:
            x: Input tensor with shape ``[..., L, D]``.
            e: Modulation tensor with shape ``[..., 6, D]``.
            cache: KV cache container for this block.
            rope_freqs: RoPE frequencies of shape
                ``[L, 1, 1, head_dim // 2]``.
            plucker_embedding: Optional camera-control Plücker embedding
                of shape ``[..., L, D]``. ``None`` disables the camera
                modulation (pass-through).

        Returns:
            Updated hidden states with shape ``[..., L, D]``.
        """
        e_chunks = (self.modulation + e).chunk(6, dim=-2)

        y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
        y = self.self_attn(
            y,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        )
        x = x + (y * e_chunks[2])

        # camera control
        camera_hidden_states = self.cam_injector_layer2(
            F.silu(self.cam_injector_layer1(plucker_embedding))
        )
        camera_hidden_states = camera_hidden_states + plucker_embedding
        camera_scale = self.cam_scale_layer(camera_hidden_states)
        camera_shift = self.cam_shift_layer(camera_hidden_states)
        x = (1.0 + camera_scale) * x + camera_shift

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(y)
        x = x + (y * e_chunks[5])
        return x
