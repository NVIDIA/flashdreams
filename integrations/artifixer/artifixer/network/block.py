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

Mirrors ``ArtifixerTransformerBlock`` in the ArtiFixer reference
implementation. Per-block extensions on top of :class:`Block`:

  * **Opacity + camera-ray MLPs.** Two ``nn.Linear`` heads project
    per-token opacity and Plucker-camera-ray features into the
    transformer hidden size; their outputs are added to the AdaLN-normed
    hidden states *before* self-attention. Both heads are zero-initialized
    so the wrapped block is a no-op extension of base Wan behavior at
    load time.

  * **Neighbor cross-attention.** ``cross_attn`` is :class:`ArtifixerCrossAttention`,
    which carries ``add_k_proj`` / ``add_v_proj`` / ``norm_added_k`` and a
    separate ``attn_op_neighbor`` attention op for the neighbor-frame KV
    branch. When the neighbor cache is unset / PRoPE modules are absent
    the forward path is identical to ``CrossAttention.forward``.
"""

from __future__ import annotations

from typing import Any

from artifixer.network.cross_attn import ArtifixerCrossAttention
from torch import Tensor, nn
from torch.nn import functional as F

from flashdreams.recipes.wan.transformer.impl.modules import Block, BlockCache


def _layer_norm_fp32(x: Tensor, norm: nn.LayerNorm) -> Tensor:
    """Run ``norm`` with fp32 input AND fp32 weight/bias, then cast back.

    Mirrors diffusers' ``FP32LayerNorm`` which the reference
    ``WanTransformerBlock`` norms use under the hood: regardless of the
    surrounding bf16 cast, the norm itself accumulates in fp32 to keep
    the mean/std numerically stable. Plain
    ``nn.LayerNorm(elementwise_affine=True)`` stores weight/bias in the
    model dtype (bf16 after ``.to(bf16)``), so calling it with an fp32
    input raises ``expected scalar type Float but found BFloat16``. The
    work-around is to call ``F.layer_norm`` directly with promoted
    parameters.
    """
    weight = norm.weight.float() if norm.weight is not None else None
    bias = norm.bias.float() if norm.bias is not None else None
    return F.layer_norm(x.float(), norm.normalized_shape, weight, bias, norm.eps)


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
        # neighbor-frame KV branch.
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
        # behavior at load time.
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

        Extras:

          - ``opacity_extra`` / ``camera_extra``: per-token opacity and
            Plucker-camera-ray features added to the AdaLN-normed hidden
            states before self-attention.
          - ``prope_src`` / ``prope_tgt`` / ``ignore_neighbors``: forwarded
            to :class:`ArtifixerCrossAttention.forward` to drive the PRoPE
            neighbor branch. The actual neighbor K/V cache lives on the
            cross_attn module itself (set via
            ``cross_attn.initialize_neighbor_cache``).

        When every extra is ``None`` / ``False`` and the cross_attn's
        neighbor cache is unset, this is identical to :meth:`Block.forward`.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "before running the forward pass"
        )
        # Mirror the ArtiFixer reference's ``ArtifixerTransformerBlock.forward``
        # and promote the per-block AdaLN modulation, RMS-norms, and residual
        # adds to fp32 before casting back to the input dtype. The base
        # ``Block.forward`` keeps everything in ``x.dtype`` (typically bf16)
        # which is fine in isolation, but with 30 blocks each contributing
        # ~1 dB of bf16 noise the cross-backend PSNR drifts from 51 dB at
        # block 0 down to 28 dB at block 29. The fp32 promotion below
        # mirrors the reference:
        #
        #   * modulation chunking: ``(scale_shift_table + temb).float().chunk(6)``
        #   * self-attn pre-norm + AdaLN
        #   * self-attn residual (x + attn * gate)
        #   * cross-attn pre-norm
        #   * FFN pre-norm + AdaLN
        #   * FFN residual (x + ffn.float() * gate) -- note the reference
        #     also promotes ``ff_output`` to fp32 inside the residual.
        #
        # Cost: a handful of extra casts per block. Parity gain: closes
        # the per-block drift so the cross-backend per-call PSNR stays
        # >50 dB through all 30 layers. The flashdreams ``Block``
        # bf16-throughout path remains the default; this override only
        # kicks in when ``ArtifixerBlock`` is used (the artifixer recipe).
        e_chunks = (self.modulation.float() + e.float()).chunk(6, dim=-2)
        x_dtype = x.dtype

        # ``norm1`` / ``norm2`` in the base Wan ``Block`` are
        # ``elementwise_affine=False`` (no weight/bias) so calling them
        # with an fp32 input "just works" -- they return fp32. ``norm3``
        # has ``elementwise_affine=True`` and stores its weight/bias in the
        # model dtype, so we route through ``_layer_norm_fp32`` to also
        # promote the weight/bias to fp32 (matches diffusers'
        # FP32LayerNorm used by the reference ``WanTransformerBlock``).
        y = (self.norm1(x.float()) * (1 + e_chunks[1]) + e_chunks[0]).to(x_dtype)
        if opacity_extra is not None:
            y = y + self.opacity_embedding(opacity_extra)
        if camera_extra is not None:
            y = y + self.camera_embedding(camera_extra)
        y = self.self_attn(
            y,
            rope_freqs=rope_freqs,
            kv_cache=cache.self_attn,
        )
        x = (x.float() + y * e_chunks[2]).to(x_dtype)

        # Cross-attn pre-norm in fp32 (mirrors the reference). The
        # post-cross-attn residual is NOT promoted in the reference so we
        # keep that in the input dtype here too.
        x = x + self.cross_attn(
            _layer_norm_fp32(x, self.norm3).to(x_dtype),
            kv_cache=cache.cross_attn,
            prope_src=prope_src,
            prope_tgt=prope_tgt,
            ignore_neighbors=ignore_neighbors,
        )
        y = (self.norm2(x.float()) * (1 + e_chunks[4]) + e_chunks[3]).to(x_dtype)
        y = self.ffn(y)
        # FFN residual with ff_output also promoted (matches the reference).
        x = (x.float() + y.float() * e_chunks[5]).to(x_dtype)
        return x
