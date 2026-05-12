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

"""ArtiFixer cross-attention with a neighbor-frame KV bank.

Adds a third KV pathway (parallel to text and optional I2V image) for
attending over VAE-encoded neighbor frames. Module naming mirrors the
diffusers ``WanAttention`` extension used in dreamfix
(``model_training/net/transformer.py`` L670-L699):

  - ``add_k_proj`` / ``add_v_proj`` — Linear projections from latent
    neighbor context to K / V
  - ``norm_added_k`` — RMSNorm on K (matches ``norm_k`` shape)
  - ``attn_op_neighbor`` — separate RingAttention op so the neighbor
    branch can later carry PRoPE-transformed q/k without entangling
    the text branch

Phase 2.2 only adds these parameters. Forward currently inherits from
``CrossAttention`` and does *not* yet wire the neighbor branch: callers
that do not pass a neighbor context observe identical behavior. The
wiring lands with the pipeline changes in Phase 3, and PRoPE q/k/v/o
transforms layer on in Phase 2.3.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from flashdreams.core.attention.ring import RingAttention
from flashdreams.recipes.wan.transformer.impl.modules import CrossAttention


class ArtifixerCrossAttention(CrossAttention):
    """Cross-attention extended with a neighbor-frame KV branch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Parameter names match dreamfix (WanAttention.add_k_proj /
        # add_v_proj / norm_added_k) so loading a merged ArtiFixer DMD
        # checkpoint only needs the diffusers ``attn2 -> cross_attn``
        # regex remap (added in Phase 5), not a per-key rename.
        self.add_k_proj = nn.Linear(self.context_dim, self.inner_dim, bias=True)
        self.add_v_proj = nn.Linear(self.context_dim, self.inner_dim, bias=True)
        self.norm_added_k = nn.RMSNorm(self.inner_dim, eps=self.eps)

        # Zero-init add_v_proj only (matches dreamfix transformer.py
        # L687-L688). Because attention is softmax(Q K^T) @ V, V=0 forces
        # the neighbor branch contribution to zero at load time even if
        # add_k_proj is non-zero, so the wrapped block is a no-op
        # extension of base Wan behavior.
        nn.init.zeros_(self.add_v_proj.weight)
        nn.init.zeros_(self.add_v_proj.bias)

        self.attn_op_neighbor = RingAttention(qkv_format="bshd", backend="cudnn")

    def set_context_parallel_group(self, cp_group: Any) -> None:
        """Set CP group on every attention op, including the neighbor branch."""
        super().set_context_parallel_group(cp_group)
        self.attn_op_neighbor.set_context_parallel_group(cp_group=cp_group)

    def compute_kv_neighbor(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project neighbor ``context`` into K/V tensors.

        Args:
            context: Neighbor context, shape ``[..., L_neighbor, context_dim]``.

        Returns:
            ``(k, v)`` reshaped to ``[batch_size, L, n_heads, head_dim]``.
        """
        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        L, _ = context.shape[-2:]
        n, d = self.n_heads, self.head_dim

        k = self.norm_added_k(self.add_k_proj(context)).reshape(batch_size, L, n, d)
        v = self.add_v_proj(context).reshape(batch_size, L, n, d)
        return k, v
