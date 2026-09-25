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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from flashdreams.accelerated.multi_head_attention.triton import (
    flash_attention_2,
    flash_attention_2_tma,
    is_tma_flash_attention_supported,
)
from flashdreams.accelerated.multi_head_attention.triton.scaled_fp8_attention import (
    scaled_fp8_attention,
)
from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import Granularity, quantize
from flashdreams.core.attention import BlockKVCache
from flashdreams.core.attention.rope import apply_rope_freqs
from flashdreams.recipes.wan.transformer.impl.modules import (
    Block,
    BlockCache,
    SelfAttention,
)


class UpstreamRMSNorm(nn.RMSNorm):
    """Wan RMSNorm with the checkpoint reference's FP32 reduction order."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize in FP32, round to the input dtype, then apply weight."""
        assert self.eps is not None
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=x.dtype) * self.weight


def _rowwise_fp8_linear(
    projection: QuantizedNonPersistentLinear,
    x: Tensor,
) -> Tensor:
    """Apply ``projection`` with compile-fusible rowwise FP8 quantization."""
    if x.numel() == 0:
        return projection(x, Granularity.SLICE, out_dtype=x.dtype)
    quantized, scale = quantize(
        x,
        projection.dtype,
        Granularity.SLICE,
        axis=-1,
        use_triton=False,
    )
    return projection(quantized, scale, out_dtype=x.dtype)


@torch.compiler.disable
def _call_fp8_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    use_tma: bool,
) -> Tensor:
    """Run direct FP8 attention, preferring TMA when supported."""
    query_fp8 = query.to(torch.float8_e4m3fn)
    key_fp8 = key.to(torch.float8_e4m3fn)
    value_fp8 = value.to(torch.float8_e4m3fn)
    attention = (
        flash_attention_2_tma
        if use_tma and is_tma_flash_attention_supported(query_fp8, key_fp8, value_fp8)
        else flash_attention_2
    )
    return attention(query_fp8, key_fp8, value_fp8, output_dtype=query.dtype)


class OptimizedSelfAttention(SelfAttention):
    """Wan self-attention with one fused FP8 QKV projection."""

    fused_qkv: QuantizedNonPersistentLinear
    quantized_output: QuantizedNonPersistentLinear

    def __init__(
        self,
        dim: int,
        n_heads: int,
        eps: float,
        *,
        attention_backend: Literal["fp8_tma", "scaled_fp8"] = "scaled_fp8",
        use_tma: bool = True,
    ) -> None:
        super().__init__(
            query_dim=dim,
            n_heads=n_heads,
            head_dim=dim // n_heads,
            eps=eps,
        )
        self.attention_backend = attention_backend
        self.use_tma = use_tma
        self._refresh_derived_weights()
        self.register_load_state_dict_post_hook(self._refresh_derived_weights)

    @torch.no_grad()
    def _refresh_derived_weights(self, *args: object) -> None:
        """Rebuild nonpersistent fused weights from checkpoint-native linears."""
        del args
        biases = (self.q.bias, self.k.bias, self.v.bias)
        self.fused_qkv = QuantizedNonPersistentLinear(
            torch.cat((self.q.weight, self.k.weight, self.v.weight), dim=0),
            torch.cat(biases, dim=0)
            if all(bias is not None for bias in biases)
            else None,
            WeightGranularity.PER_OUT_CHANNEL,
            torch.float8_e4m3fn,
        )
        self.quantized_output = QuantizedNonPersistentLinear(
            self.o.weight,
            self.o.bias,
            WeightGranularity.PER_OUT_CHANNEL,
            torch.float8_e4m3fn,
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> "OptimizedSelfAttention":
        """Move canonical tensors, then rebuild derived FP8 projections."""
        module = super()._apply(fn, recurse=recurse)
        self._refresh_derived_weights()
        return module

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Reject context parallelism, which the fused projection does not support."""
        if cp_group is not None:
            raise NotImplementedError(
                "Fused Lingbot attention does not support context parallelism"
            )

    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Project Q/K/V together, update the cache, and apply attention."""
        batch_shape = x.shape[:-2]
        sequence_length = x.shape[-2]
        qkv = _rowwise_fp8_linear(self.fused_qkv, x)
        q, k, v = qkv.chunk(3, dim=-1)
        head_shape = (-1, sequence_length, self.n_heads, self.head_dim)
        q = self.norm_q(q).reshape(head_shape)
        k = self.norm_k(k).reshape(head_shape)
        v = v.reshape(head_shape)
        q = apply_rope_freqs(q, rope_freqs, interleaved=True)
        k = apply_rope_freqs(k, rope_freqs, interleaved=True)
        kv_cache.update(k, v)
        if self.attention_backend == "fp8_tma":
            output = _call_fp8_attention(
                q,
                kv_cache.cached_k(),
                kv_cache.cached_v(),
                use_tma=self.use_tma,
            )
        else:
            output = scaled_fp8_attention(
                q,
                kv_cache.cached_k(),
                kv_cache.cached_v(),
                output_dtype=q.dtype,
                use_tma=self.use_tma,
            )
        output = output.reshape(batch_shape + (sequence_length, self.inner_dim))
        return _rowwise_fp8_linear(self.quantized_output, output)


@dataclass
class CamCtrlBlockCache(BlockCache):
    """Per-block KV and camera-modulation cache."""

    camera_scale: Tensor | None = None
    """Cached camera scale for the current autoregressive chunk."""

    camera_shift: Tensor | None = None
    """Cached camera shift for the current autoregressive chunk."""


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
        self_attention_backend: Literal["wan", "fp8_tma", "scaled_fp8"] = "wan",
        self_attention_use_tma: bool = True,
    ) -> None:
        super().__init__(
            dim=dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            cp_method=cp_method,
        )
        if self_attention_backend != "wan":
            self.self_attn = OptimizedSelfAttention(
                dim,
                num_heads,
                eps,
                attention_backend=self_attention_backend,
                use_tma=self_attention_use_tma,
            )
        # The checkpoint's Wan implementation uses an explicit FP32 square/mean
        # reduction, rounds the normalized values back to the activation dtype,
        # and only then applies the learned scale. ``nn.RMSNorm`` is close but
        # not numerically equivalent; the per-layer difference compounds over
        # long autoregressive rollouts.
        self.self_attn.norm_q = UpstreamRMSNorm(dim, eps)
        self.self_attn.norm_k = UpstreamRMSNorm(dim, eps)
        self.cross_attn.norm_q = UpstreamRMSNorm(dim, eps)
        self.cross_attn.norm_k = UpstreamRMSNorm(dim, eps)
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context_text: Tensor,
        context_img: Tensor | None = None,
    ) -> CamCtrlBlockCache:
        """Initialize KV and camera-modulation caches for this block."""
        cache = super().initialize_cache(
            chunk_size,
            window_size,
            sink_size,
            context_text,
            context_img,
        )
        return CamCtrlBlockCache(
            self_attn=cache.self_attn,
            cross_attn=cache.cross_attn,
        )

    def prepare_camera_cache(
        self,
        plucker_embedding: Tensor,
        cache: CamCtrlBlockCache,
    ) -> None:
        """Refresh camera modulation while preserving allocated buffer addresses."""
        camera_hidden_states = self.cam_injector_layer2(
            F.silu(self.cam_injector_layer1(plucker_embedding))
        )
        camera_hidden_states = camera_hidden_states + plucker_embedding
        camera_scale = self.cam_scale_layer(camera_hidden_states)
        camera_shift = self.cam_shift_layer(camera_hidden_states)

        if cache.camera_scale is None:
            cache.camera_scale = camera_scale
            cache.camera_shift = camera_shift
            return

        assert cache.camera_shift is not None
        assert cache.camera_scale.shape == camera_scale.shape
        assert cache.camera_shift.shape == camera_shift.shape
        cache.camera_scale.copy_(camera_scale)
        cache.camera_shift.copy_(camera_shift)

    def forward(
        self,
        x: Tensor,
        e: Tensor,
        cache: CamCtrlBlockCache,
        rope_freqs: Tensor,
    ) -> Tensor:
        """Run one transformer block update.

        Args:
            x: Input tensor with shape ``[..., L, D]``.
            e: Modulation tensor with shape ``[..., 6, D]``.
            cache: KV and camera-modulation cache for this block.
            rope_freqs: RoPE frequencies of shape
                ``[L, 1, 1, head_dim // 2]``.

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

        assert cache.camera_scale is not None and cache.camera_shift is not None, (
            "prepare_camera_cache must run before CamCtrlBlock.forward"
        )
        x = (1.0 + cache.camera_scale) * x + cache.camera_shift

        x = x + self.cross_attn(
            self.norm3(x),
            kv_cache=cache.cross_attn,
        )
        y = self.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]
        y = self.ffn(y)
        x = x + (y * e_chunks[5])
        return x
