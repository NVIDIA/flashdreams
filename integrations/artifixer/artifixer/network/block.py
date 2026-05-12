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

"""ArtiFixer transformer block with opacity + camera + neighbor extensions.

Mirrors ``ArtifixerTransformerBlock`` in the dreamfix reference
(``model_training/net/transformer.py`` L617-L767). Per-block extensions on
top of :class:`Block`:

  * **Phase 2.1** — opacity + camera-ray MLPs. Two ``nn.Linear`` heads
    project per-token opacity and Plucker-camera-ray features into the
    transformer hidden size; their outputs are added to the AdaLN-normed
    hidden states *before* self-attention. Both heads are zero-initialized.

  * **Phase 2.2** — ``cross_attn`` is replaced with
    :class:`ArtifixerCrossAttention`, which carries ``add_k_proj`` /
    ``add_v_proj`` / ``norm_added_k`` and a separate ``attn_op_neighbor``
    RingAttention op for a future neighbor-frame KV branch. The forward
    path is unchanged (parent ``CrossAttention.forward`` is called); the
    pipeline-level neighbor wiring lands in Phase 3.
"""

from __future__ import annotations

from typing import Any

import torch
from artifixer.network.cross_attn import ArtifixerCrossAttention
from torch import Tensor, nn

from flashdreams.recipes.wan.transformer.impl.modules import Block, BlockCache


class ArtifixerBlock(Block):
    """Wan transformer block extended with opacity + camera-ray MLPs."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        opacity_embedding_dim: int,
        camera_embedding_dim: int,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        i2v: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            i2v=i2v,
        )

        # Replace the inherited cross_attn with the ArtiFixer variant that
        # also carries add_k_proj / add_v_proj / norm_added_k for the
        # future neighbor-frame KV branch (Phase 3 wiring).
        self.cross_attn = ArtifixerCrossAttention(
            query_dim=dim,
            n_heads=num_heads,
            head_dim=dim // num_heads,
            i2v=i2v,
            eps=eps,
        )

        self.opacity_embedding = nn.Linear(opacity_embedding_dim, dim, bias=True)
        self.camera_embedding = nn.Linear(camera_embedding_dim, dim, bias=True)

        # Zero-init so the wrapped block is a no-op extension of base Wan
        # behavior at load time. Matches dreamfix transformer.py L637-651.
        nn.init.zeros_(self.opacity_embedding.weight)
        nn.init.zeros_(self.opacity_embedding.bias)
        nn.init.zeros_(self.camera_embedding.weight)
        nn.init.zeros_(self.camera_embedding.bias)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        e: Tensor,
        cache: BlockCache,
        rope_freqs: Tensor,
        opacity_extra: Tensor | None = None,
        camera_extra: Tensor | None = None,
        prope_src: Any = None,
        prope_tgt: Any = None,
        ignore_neighbors: bool = False,
    ) -> Tensor:
        """Run one transformer block update with ArtiFixer conditioning.

        Extras (Phase 2.1 + 2.4):

          - ``opacity_extra`` / ``camera_extra``: per-token opacity and
            Plucker-camera-ray features added to the AdaLN-normed hidden
            states before self-attention (matches dreamfix
            ``ArtifixerTransformerBlock.forward`` L763-L767).
          - ``prope_src`` / ``prope_tgt`` / ``ignore_neighbors``: forwarded
            to :class:`ArtifixerCrossAttention.forward` to drive the PRoPE
            neighbor branch (matches dreamfix L795-L807). The actual
            neighbor K/V cache lives on the cross_attn module itself
            (set via ``cross_attn.initialize_neighbor_cache``).

        When every extra is ``None`` / ``False`` and the cross_attn's
        neighbor cache is unset, this is identical to :meth:`Block.forward`.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        e_chunks = (self.modulation + e).chunk(6, dim=-2)

        y = self.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]
        if opacity_extra is not None:
            y = y + self.opacity_embedding(opacity_extra)
        if camera_extra is not None:
            y = y + self.camera_embedding(camera_extra)
        y = self.self_attn(
            y,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        )
        x = x + (y * e_chunks[2])

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
            prope_src=prope_src,
            prope_tgt=prope_tgt,
            ignore_neighbors=ignore_neighbors,
        )
        y = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(y)
        x = x + (y * e_chunks[5])
        return x
