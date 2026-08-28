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

"""Optimized multi-head attention with fused projections and selectable SDPA."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn

from flashdreams.accelerated.common.non_persistent_linear import (
    NonPersistentLinear,
)
from flashdreams.accelerated.multi_head_attention import (
    AttentionConfig,
    AttentionType,
    MultiHeadAttention,
    QKNormScope,
    RoPEScope,
    RoPEStyle,
)
from flashdreams.accelerated.multi_head_attention.cudnn import (
    native_cudnn_fp8_sdpa,
    torch_cudnn_sdpa,
)
from flashdreams.accelerated.multi_head_attention.triton import (
    flash_attention_2,
    flash_attention_2_tma,
    is_tma_flash_attention_supported,
)
from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import (
    DTYPE_MAX,
    Granularity,
    quantize,
)
from flashdreams.core.attention import BlockKVCache
from flashdreams.core.attention.rope_kernel import apply_rotary_pos_emb


class SDPABackend(str, Enum):
    """Scaled-dot-product attention implementation."""

    CUDNN = "cudnn"
    """Use Torch cuDNN for FP16/BF16 and native cuDNN Frontend for FP8."""

    FA2 = "fa2"
    """Use Triton FlashAttention2 (FA2)."""


class QKVFusionOption(str, Enum):
    """Projection fusion policy."""

    NONE = "none"
    """Project queries, keys, and values independently."""

    FULL = "full"
    """Use one QKV GEMM; query and context feature widths must match."""

    FUSE_KV = "fuse_kv"
    """Use an independent Q GEMM and one KV GEMM, allowing unequal input widths."""


@dataclass(frozen=True, slots=True)
class QuantizationOption:
    """Attention quantization policy."""

    projection: torch.dtype | None = None
    """Q/K/V projection dtype; ``None`` preserves native precision."""

    quantized_sdpa: bool = False
    """Use unscaled FP8 e4m3 Q/K/V in scaled-dot-product attention.

    This directly casts Q, K, and V to FP8 e4m3 before calling the configured
    SDPA backend and stores the K/V cache in that dtype. The cuDNN path requires
    ``nvidia-cudnn-frontend``. FA2 also casts the
    softmax probabilities used by ``P @ V`` to FP8 e4m3. This is not
    SageAttention3 quantization: it uses no accuracy-preserving quantization
    scheme or scaling. As noted by the SageAttention3 paper, this simple approach
    can make attention inaccurate, so output accuracy is not guaranteed.
    """

    def __post_init__(self) -> None:
        """Validate the projection quantization dtype."""
        if self.projection is not None and self.projection not in DTYPE_MAX:
            raise ValueError(
                f"unsupported projection quantization dtype: {self.projection}"
            )
        if not isinstance(self.quantized_sdpa, bool):
            raise TypeError(
                f"quantized_sdpa must be a bool; got {self.quantized_sdpa!r}"
            )


@dataclass(frozen=True, slots=True)
class OptimizedImplConfig:
    """Optimized attention implementation policy."""

    qkv_fusion_option: QKVFusionOption = QKVFusionOption.FULL
    """Projection fusion policy."""

    sdpa_backend: SDPABackend = SDPABackend.CUDNN
    """Scaled-dot-product attention implementation."""

    use_tma: bool = True
    """Prefer TMA FlashAttention2 when the device and tensors support it."""

    quantization: QuantizationOption = QuantizationOption()
    """Attention quantization policy."""

    def __post_init__(self) -> None:
        """Validate optimized implementation policy values."""
        if not isinstance(self.qkv_fusion_option, QKVFusionOption):
            raise TypeError(
                "qkv_fusion_option must be a QKVFusionOption; "
                f"got {self.qkv_fusion_option!r}"
            )
        if not isinstance(self.quantization, QuantizationOption):
            raise TypeError(
                f"quantization must be a QuantizationOption; got {self.quantization!r}"
            )
        if not isinstance(self.sdpa_backend, SDPABackend):
            raise TypeError(
                f"sdpa_backend must be an SDPABackend; got {self.sdpa_backend!r}"
            )
        if not isinstance(self.use_tma, bool):
            raise TypeError(f"use_tma must be a bool; got {self.use_tma!r}")


class OptimizedMultiHeadAttention(MultiHeadAttention[BlockKVCache]):
    """Provide inference-only streaming self- and static cross-attention.

    Shape comments use ``B`` for the product of all leading batch dimensions,
    ``L`` for the current query/chunk length, ``S`` for visible cached context,
    ``H`` for the number of heads, ``D`` for the head dimension, ``Q`` for
    ``query_dim``, and ``C`` for ``context_dim``.

    Full-fusion self-attention produces Q/K/V in one GEMM before PyTorch Q/K
    normalization, the shared RoPE kernel, and standard cache updates.
    Cross-attention precomputes K/V independently and reuses that static cache
    across forward calls.

    Concrete subclasses own their checkpoint-native projection and
    normalization modules and map them to the logical module properties.
    Callers own the :class:`BlockKVCache` lifecycle: call ``before_update`` before
    streaming self-attention and ``after_update`` once every block has consumed
    that chunk.
    """

    @property
    @abstractmethod
    def query_projection(self) -> nn.Linear:
        """Return the query projection module."""

    @property
    @abstractmethod
    def key_projection(self) -> nn.Linear:
        """Return the key projection module."""

    @property
    @abstractmethod
    def value_projection(self) -> nn.Linear:
        """Return the value projection module."""

    @property
    @abstractmethod
    def output_projection(self) -> nn.Linear:
        """Return the attention output projection module."""

    @property
    @abstractmethod
    def query_norm(self) -> nn.Module:
        """Return the query normalization module or identity."""

    @property
    @abstractmethod
    def key_norm(self) -> nn.Module:
        """Return the key normalization module or identity."""

    optimized_impl_config: OptimizedImplConfig
    """Projection and attention backend policy."""

    fused_qkv: NonPersistentLinear | None
    """Nonpersistent full-fusion Q/K/V projection."""

    fused_kv: NonPersistentLinear | None
    """Nonpersistent fused K/V projection."""

    quantized_query_projection: QuantizedNonPersistentLinear | None
    """Nonpersistent quantized query projection."""

    quantized_key_projection: QuantizedNonPersistentLinear | None
    """Nonpersistent quantized key projection."""

    quantized_value_projection: QuantizedNonPersistentLinear | None
    """Nonpersistent quantized value projection."""

    _validated_cuda_device_index: int | None
    """CUDA device whose compute capability has passed validation."""

    def __init__(
        self,
        attention_type: AttentionType,
        attention_config: AttentionConfig,
        optimized_impl_config: OptimizedImplConfig,
    ) -> None:
        """Initialize shared optimized attention policies.

        Args:
            attention_type: Select self-attention or static cross-attention.
            attention_config: Shared geometry, normalization, and rotary policy.
            optimized_impl_config: Projection and attention backend policy.

        Raises:
            TypeError: ``optimized_impl_config`` has the wrong type.
            ValueError: A fusion or head dimension is unsupported.
        """
        super().__init__(attention_type, attention_config)
        if not isinstance(optimized_impl_config, OptimizedImplConfig):
            raise TypeError(
                "optimized_impl_config must be an OptimizedImplConfig; "
                f"got {optimized_impl_config!r}"
            )
        qkv_fusion_option = optimized_impl_config.qkv_fusion_option
        if (
            qkv_fusion_option is QKVFusionOption.FULL
            and self.attention_config.query_dim != self.attention_config.context_dim
        ):
            raise ValueError(
                "full QKV fusion requires query_dim to equal context_dim; "
                f"got {self.attention_config.query_dim} and {self.attention_config.context_dim}"
            )
        if not (
            16 <= self.attention_config.head_dim <= 256
            and self.attention_config.head_dim & (self.attention_config.head_dim - 1)
            == 0
        ):
            raise ValueError(
                "accelerated attention requires a power-of-two head_dim in [16, 256]; "
                f"got {self.attention_config.head_dim}"
            )

        self.optimized_impl_config = optimized_impl_config
        self.qkv_fusion_option = optimized_impl_config.qkv_fusion_option
        self.sdpa_backend = optimized_impl_config.sdpa_backend
        self.use_tma = optimized_impl_config.use_tma
        self._validated_cuda_device_index = None

    # ---------------------- Initialization ---------------------- #

    def _initialize_derived_weights(self) -> None:
        """Build fused execution modules after concrete checkpoint fields exist.

        Concrete implementations call this after assigning their checkpoint
        fields. The logical accessors are valid before fused projections or load
        hooks read any projection or normalization module.
        """
        self.fused_qkv = None
        self.fused_kv = None
        self.quantized_query_projection = None
        self.quantized_key_projection = None
        self.quantized_value_projection = None
        self._refresh_derived_weights()
        self.register_load_state_dict_post_hook(self._refresh_derived_weights)

    @staticmethod
    def _new_quantized_projection(
        weight: Tensor,
        bias: Tensor | None,
        dtype: torch.dtype,
    ) -> QuantizedNonPersistentLinear:
        """Build a per-output-channel quantized projection from checkpoint tensors."""
        return QuantizedNonPersistentLinear(
            weight.detach().contiguous(),
            None if bias is None else bias.detach(),
            WeightGranularity.PER_OUT_CHANNEL,
            dtype,
        )

    @torch.no_grad()
    def _refresh_derived_weights(self, *args: object) -> None:
        """Rebuild fused projection modules from checkpoint parameters.

        Raises:
            ValueError: Query, key, and value projections mix bias policies.
        """
        del args
        self.fused_qkv = None
        self.fused_kv = None
        self.quantized_query_projection = None
        self.quantized_key_projection = None
        self.quantized_value_projection = None

        projection_dtype = self.optimized_impl_config.quantization.projection
        projection_biases = (
            self.query_projection.bias,
            self.key_projection.bias,
            self.value_projection.bias,
        )
        present_biases = tuple(bias for bias in projection_biases if bias is not None)
        if present_biases and len(present_biases) != len(projection_biases):
            raise ValueError(
                "query, key, and value projections must either all have biases "
                "or all omit biases"
            )

        if projection_dtype is not None:
            self.quantized_query_projection = self._new_quantized_projection(
                self.query_projection.weight,
                self.query_projection.bias,
                projection_dtype,
            )
            if self.qkv_fusion_option is QKVFusionOption.NONE:
                self.quantized_key_projection = self._new_quantized_projection(
                    self.key_projection.weight,
                    self.key_projection.bias,
                    projection_dtype,
                )
                self.quantized_value_projection = self._new_quantized_projection(
                    self.value_projection.weight,
                    self.value_projection.bias,
                    projection_dtype,
                )

        if self.qkv_fusion_option is QKVFusionOption.FULL:
            fused_weight = (
                torch.cat(
                    (
                        self.query_projection.weight,
                        self.key_projection.weight,
                        self.value_projection.weight,
                    ),
                    dim=0,
                )
                .detach()
                .contiguous()
            )
            fused_bias = (
                torch.cat(present_biases, dim=0).detach() if present_biases else None
            )
            fused_kv_weight = fused_weight[self.attention_config.inner_dim :]
            fused_kv_bias = (
                None
                if fused_bias is None
                else fused_bias[self.attention_config.inner_dim :]
            )
            if projection_dtype is None:
                self.fused_qkv = NonPersistentLinear(fused_weight, fused_bias)
                self.fused_kv = NonPersistentLinear(fused_kv_weight, fused_kv_bias)
            else:
                self.fused_qkv = self._new_quantized_projection(
                    fused_weight, fused_bias, projection_dtype
                )
                self.fused_kv = self._new_quantized_projection(
                    fused_kv_weight, fused_kv_bias, projection_dtype
                )
        elif self.qkv_fusion_option is QKVFusionOption.FUSE_KV:
            fused_weight = (
                torch.cat(
                    (self.key_projection.weight, self.value_projection.weight), dim=0
                )
                .detach()
                .contiguous()
            )
            fused_bias = (
                torch.cat(present_biases[1:], dim=0).detach()
                if present_biases
                else None
            )
            if projection_dtype is None:
                self.fused_kv = NonPersistentLinear(fused_weight, fused_bias)
            else:
                self.fused_kv = self._new_quantized_projection(
                    fused_weight, fused_bias, projection_dtype
                )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> OptimizedMultiHeadAttention:
        """Transform parameters and rebuild fused modules on their final device.

        Args:
            fn: Tensor transformation applied by :class:`torch.nn.Module`.
            recurse: Apply ``fn`` recursively to child modules.

        Returns:
            This module with fused projection modules refreshed.
        """
        # Move/cast canonical parameters first, then regenerate fused projections.
        module = super()._apply(fn, recurse=recurse)
        self._refresh_derived_weights()
        return module

    # ------------------------------------------------------------ #
    #                        Public Methods                        #
    # ------------------------------------------------------------ #

    def allocate_kv_cache(
        self,
        batch_size: int,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> BlockKVCache:
        """Allocate a rolling cache for native or quantized SDPA.

        Args:
            batch_size: Flattened batch size ``B``.
            chunk_size: Number of current tokens ``L`` written per update.
            window_size: Number of rolling context tokens retained after the sink.
            sink_size: Number of initial context tokens that are never evicted.
            device: Device on which to allocate K/V storage.
            dtype: FP16 or BF16 activation dtype. Quantized SDPA stores the cache
                in FP8 e4m3 instead.

        Returns:
            Block cache with K/V storage shaped
            ``[B, sink_size + window_size, H, D]``.

        """
        # Keep ``[B, S, H, D]`` as the public cache shape: ``BlockKVCache``
        # rolls and slices axis 1, and both attention backends accept that logical
        # order.
        cache_shape = (
            batch_size,
            sink_size + window_size,
            self.attention_config.n_heads,
            self.attention_config.head_dim,
        )
        self._validate_cuda_device(device)
        return BlockKVCache(
            k_shape=cache_shape,
            v_shape=cache_shape,
            seq_dim=1,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=(
                torch.float8_e4m3fn
                if self.optimized_impl_config.quantization.quantized_sdpa
                else dtype
            ),
        )

    @torch.no_grad()
    def compute_kv(
        self,
        context: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Project complete context into a reusable static K/V cache.

        Args:
            context: Context tokens shaped ``[..., S, C]``.
            rope_freqs: Optional key rotations shaped ``[S, 1, 1, D]``.
                Applied only by before-cache RoPE; after-cache RoPE stores
                unrotated keys. Ignored when ``rope_config`` is ``None``.

        Returns:
            Filled cache with logical K/V shape ``[B, S, H, D]``. Its sequence
            length and window both equal S, so subsequent forward calls read all
            context.
        """
        key, value = self._project_kv(context)
        if (
            self.attention_config.rope_config is not None
            and self.attention_config.rope_config.scope is RoPEScope.BEFORE_KV_CACHE
            and rope_freqs is not None
        ):
            # Position affects key directions used in Q·K; values are never rotated.
            key = self._apply_rope(key, rope_freqs)

        if self.optimized_impl_config.quantization.quantized_sdpa:
            key = key.to(torch.float8_e4m3fn)
            value = value.to(torch.float8_e4m3fn)

        return BlockKVCache.from_tensor(key, value, seq_dim=1)

    @torch.no_grad()
    def forward(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None = None,
    ) -> Tensor:
        """Apply self- or cross-attention using the configured cache lifecycle.

        Self-attention updates the prepared rolling cache and computes Q in one
        backend-owned branch. Cross-attention computes Q while leaving its
        precomputed static cache unchanged.

        Args:
            x: Query tokens shaped ``[..., L, Q]``.
            kv_cache: Prepared rolling cache for self-attention or precomputed
                static cache for cross-attention.
            rope_freqs: Optional positional data. Before-cache RoPE expects the
                current chunk. After-cache RoPE expects positions covering the
                query and visible cache. Ignored when ``rope_config`` is
                ``None``.

        Returns:
            Output-projected tokens with the same shape and dtype as ``x``.
        """
        query_rope_freqs, key_rope_freqs = self._slice_rope_freqs(
            rope_freqs, kv_cache, x.shape[-2]
        )
        if self.attention_type is AttentionType.SELF_ATTENTION:
            query = self._update_kv_and_compute_query(x, kv_cache, query_rope_freqs)
        else:
            query = self._compute_query(x, query_rope_freqs)
            self._validate_cache(kv_cache, x)

        # ``cached_k/v`` expose only the valid prefix while a rolling cache fills,
        # and the complete fixed-size buffer after it reaches steady state.
        key = kv_cache.cached_k()
        if (
            self.attention_config.rope_config is not None
            and self.attention_config.rope_config.scope is RoPEScope.AFTER_KV_CACHE
            and key_rope_freqs is not None
        ):
            # The shared RoPE kernel is in-place; keep cache storage unrotated so
            # rolling positions can be applied again on the next attention call.
            key = self._apply_rope(key.to(x.dtype, copy=True), key_rope_freqs)
        value = kv_cache.cached_v()
        if self.optimized_impl_config.quantization.quantized_sdpa:
            query = query.to(torch.float8_e4m3fn)
            key = key.to(torch.float8_e4m3fn)
            value = value.to(torch.float8_e4m3fn)
        output = self._attention(
            query,
            key,
            value,
            output_dtype=x.dtype,
        )
        sequence_length = x.shape[-2]
        output = output.reshape(-1, sequence_length, self.attention_config.inner_dim)
        output = self._project_output(output)
        return output.reshape(
            x.shape[:-2] + (sequence_length, self.attention_config.query_dim)
        )

    # ------------------------------------------------------------ #
    #                        Private Method                        #
    # ------------------------------------------------------------ #

    # ------------------ Core Attention Methods ------------------ #

    def _slice_rope_freqs(
        self,
        rope_freqs: Tensor | None,
        kv_cache: BlockKVCache,
        query_length: int,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Select query and key rotations for the configured cache scope.

        Args:
            rope_freqs: Current-chunk or query/cache-relative rotation angles.
            kv_cache: Cache whose visible and current write ranges select angles.
            query_length: Number of query tokens.

        Returns:
            Query and visible-key rotation slices, or two ``None`` values when
            rotary embeddings are disabled.
        """
        if self.attention_config.rope_config is None or rope_freqs is None:
            return None, None
        if self.attention_config.rope_config.scope is RoPEScope.BEFORE_KV_CACHE:
            return rope_freqs, rope_freqs

        key_rope_freqs = rope_freqs[: kv_cache.size]
        if self.attention_type is AttentionType.CROSS_ATTENTION:
            return rope_freqs[:query_length], key_rope_freqs
        write_end = kv_cache.write_end
        write_start = write_end - query_length
        return rope_freqs[write_start:write_end], key_rope_freqs

    def _compute_query(
        self,
        query: Tensor,
        rope_freqs: Tensor | None,
        prequantized_input: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Project, normalize, and optionally rotate query tokens.

        Args:
            query: Full-precision query tokens.
            rope_freqs: Optional query rotations.
            prequantized_input: Quantized ``query`` and its scale, or ``None``
                to quantize a separate query projection in this call.

        Returns:
            Processed query tensor in token-major head layout.
        """
        if self.attention_config.rope_config is None:
            rope_freqs = None
        query = self._project_query(query, prequantized_input)
        if self.attention_config.rope_config is not None and rope_freqs is not None:
            query = self._apply_rope(query, rope_freqs)
        return query

    def _update_kv_and_compute_query(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None,
    ) -> Tensor:
        """Update rolling K/V and return the processed current query."""
        if self.attention_config.rope_config is None:
            rope_freqs = None
        if self.qkv_fusion_option is QKVFusionOption.FULL:
            self._validate_fused_update_inputs(x, kv_cache, rope_freqs)
            sequence_length = x.shape[-2]
            x_flat = x.reshape(-1, sequence_length, self.attention_config.query_dim)
            query, key, value = self._project_qkv(x_flat)
            query = self._apply_qk_norm(query, self.query_norm)
            key = self._apply_qk_norm(key, self.key_norm)
            if rope_freqs is not None:
                query = self._apply_rope(query, rope_freqs)
                if (
                    self.attention_config.rope_config is not None
                    and self.attention_config.rope_config.scope
                    is RoPEScope.BEFORE_KV_CACHE
                ):
                    key = self._apply_rope(key, rope_freqs)
            if self.optimized_impl_config.quantization.quantized_sdpa:
                key = key.to(torch.float8_e4m3fn)
                value = value.to(torch.float8_e4m3fn)
            kv_cache.update(key, value)
            return query

        assert self.attention_config.context_dim is not None
        self._validate_tokens(x, self.attention_config.context_dim, "context")
        self._validate_cache(kv_cache, x)
        if kv_cache._curr_chunk_idx is None:
            raise RuntimeError("call kv_cache.before_update() before attention")
        if x.shape[-2] != kv_cache.chunk_size:
            raise ValueError(
                "context sequence length must equal cache "
                f"chunk_size={kv_cache.chunk_size}; got {x.shape[-2]}"
            )

        prequantized_input = None
        if self.optimized_impl_config.quantization.projection is not None:
            prequantized_input = self._prequantize_projection_input(x)
        query = self._compute_query(x, rope_freqs, prequantized_input)
        key, value = self._project_kv(x, prequantized_input)
        if (
            self.attention_config.rope_config is not None
            and self.attention_config.rope_config.scope is RoPEScope.BEFORE_KV_CACHE
            and rope_freqs is not None
        ):
            key = self._apply_rope(key, rope_freqs)
        if self.optimized_impl_config.quantization.quantized_sdpa:
            key = key.to(torch.float8_e4m3fn)
            value = value.to(torch.float8_e4m3fn)
        kv_cache.update(key, value)
        return query

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Apply the configured non-causal scaled-dot-product attention backend.

        Args:
            query: Processed queries with shape ``[B, L, H, D]``.
            key: Cached keys with shape ``[B, S, H, D]``.
            value: Cached values with shape ``[B, S, H, D]``.
            output_dtype: Output storage dtype; ``None`` uses ``query.dtype``.

        Returns:
            Attention output with shape ``[B, L, H, D]``.
        """
        if self.sdpa_backend is SDPABackend.CUDNN:
            # The module and Triton kernel use token-major ``[B, L/S, H, D]``.
            # PyTorch SDPA instead interprets its two middle axes as ``[H, L/S]``.
            # These transposes change only shape/stride metadata.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # PyTorch's public dispatcher rejects FP8 inputs, so use a cuDNN
            # Frontend FP8 graph for e4m3 attention.
            if query.dtype is torch.float8_e4m3fn:
                output = native_cudnn_fp8_sdpa(query, key, value)
            else:
                output = torch_cudnn_sdpa(query, key, value)

            # Restore the module-wide ``[B, L, H, D]`` contract for head merging.
            output = output.transpose(1, 2)
            return output if output_dtype is None else output.to(output_dtype)

        attention = (
            flash_attention_2_tma
            if self.use_tma and is_tma_flash_attention_supported(query, key, value)
            else flash_attention_2
        )
        if output_dtype is None:
            return attention(query, key, value)
        return attention(query, key, value, output_dtype=output_dtype)

    def _apply_rope(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        """Apply the shared RoPE kernel to token-major head features.

        Args:
            x: Projected Q or K tensor shaped ``[B, L, H, D]``.
            rope_freqs: Rotation angles shaped ``[L, 1, 1, D]``.

        Returns:
            In-place rotated tensor with the same shape and dtype as ``x``.
        """
        rope_config = self.attention_config.rope_config
        if rope_config is None:
            return x
        return apply_rotary_pos_emb(
            x,
            rope_freqs,
            interleaved=rope_config.style is RoPEStyle.INTERLEAVED,
            inplace=True,
        )

    # ------------------------ Validation ------------------------ #

    def _validate_cuda_device(self, device: torch.device | str) -> None:
        """Validate a CUDA device once before it enters the hot path.

        Args:
            device: Device used by attention inputs and cache storage.

        Raises:
            RuntimeError: The CUDA device predates Hopper.
        """
        device = torch.device(device)
        if device.type != "cuda":
            return
        device_index = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        if self._validated_cuda_device_index == device_index:
            return
        if torch.cuda.get_device_capability(device_index)[0] < 9:
            raise RuntimeError(
                "OptimizedMultiHeadAttention requires compute capability 9.0 or newer"
            )
        self._validated_cuda_device_index = device_index

    def _validate_tokens(self, x: Tensor, feature_dim: int, name: str) -> None:
        """Validate a token tensor before a CUDA projection.

        Args:
            x: Query or context tokens shaped ``[..., length, feature_dim]``.
            feature_dim: Projection input width required by the module.
            name: Argument label included in validation errors.

        Raises:
            ValueError: ``x`` lacks sequence/feature axes or has the wrong width.
            RuntimeError: ``x`` is not CUDA FP16/BF16 or the GPU predates Hopper.
        """
        if x.ndim < 2:
            raise ValueError(
                f"{name} must have shape [..., L, D]; got {tuple(x.shape)}"
            )
        if x.shape[-1] != feature_dim:
            raise ValueError(
                f"{name} feature width must equal {feature_dim}; got {x.shape[-1]}"
            )
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "OptimizedMultiHeadAttention requires CUDA FP16 or BF16 inputs"
            )
        self._validate_cuda_device(x.device)

    def _validate_cache(self, kv_cache: BlockKVCache, x: Tensor) -> None:
        """Validate a cache against query/context tokens.

        Args:
            kv_cache: Static or rolling cache with logical shape ``[B, S, H, D]``.
            x: Public tokens whose leading dimensions determine flattened batch B.

        Raises:
            ValueError: Cache rank, sequence axis, batch, head, or feature shape
                does not match this attention module.
            RuntimeError: Cache device or storage dtype does not match ``x`` and
                the configured backend.
        """
        if kv_cache.seq_dim != 1 or kv_cache._k.ndim != 4:
            raise ValueError(
                "OptimizedMultiHeadAttention requires a [B, S, H, D] cache with seq_dim=1"
            )
        # Public leading dimensions such as batch and video view collapse into
        # one B axis before projection; cached K/V must use the same flattening.
        expected_shape = (
            math.prod(x.shape[:-2]),
            self.attention_config.n_heads,
            self.attention_config.head_dim,
        )
        cache_shape = (kv_cache._k.shape[0], kv_cache._k.shape[2], kv_cache._k.shape[3])
        if cache_shape != expected_shape:
            raise ValueError(
                "cache batch, head, and feature dimensions must equal "
                f"{expected_shape}; got {cache_shape}"
            )
        if kv_cache._v.shape != kv_cache._k.shape:
            raise ValueError("Optimized attention requires identical K/V cache shapes")
        if kv_cache._k.device != x.device or kv_cache._v.device != x.device:
            raise RuntimeError("K/V cache tensors must match the input device")
        expected_dtype = (
            torch.float8_e4m3fn
            if self.optimized_impl_config.quantization.quantized_sdpa
            else x.dtype
        )
        if kv_cache._k.dtype != expected_dtype or kv_cache._v.dtype != expected_dtype:
            raise RuntimeError(f"K/V cache tensors must use {expected_dtype}")

    def _validate_fused_update_inputs(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None,
    ) -> None:
        """Validate full-fusion update inputs before cache mutation.

        Args:
            x: Current self-attention tokens, shape ``[..., L, Q]``.
            kv_cache: Prepared cache with K/V shape ``[B, S, H, D]``.
            rope_freqs: Optional current-chunk angles, shape ``[L, 1, 1, D]``.

        Raises:
            ValueError: Tensor dimensions or cache layout do not match the module.
            RuntimeError: Device, dtype, cache lifecycle, or hardware requirements
                are not satisfied.
        """
        # Validate the public token shape before flattening leading dimensions.
        if x.ndim < 2:
            raise ValueError(f"x must have shape [..., L, D]; got {tuple(x.shape)}")
        if x.shape[-1] != self.attention_config.query_dim:
            raise ValueError(
                f"x feature width must equal query_dim={self.attention_config.query_dim}; "
                f"got {x.shape[-1]}"
            )

        # The accelerated path accepts native FP16/BF16 CUDA inputs.
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "OptimizedMultiHeadAttention requires CUDA FP16 or BF16 inputs"
            )
        self._validate_cuda_device(x.device)

        # The caller prepares cache write bounds before attention. K/V storage
        # keeps logical ``[B, S, H, D]`` axes in one of two supported dense layouts.
        if kv_cache._curr_chunk_idx is None:
            raise RuntimeError("call kv_cache.before_update() before attention")
        if kv_cache.seq_dim != 1 or kv_cache._k.ndim != 4:
            raise ValueError(
                "OptimizedMultiHeadAttention requires a [B, S, H, D] cache with seq_dim=1"
            )
        if x.shape[-2] != kv_cache.chunk_size:
            raise ValueError(
                f"x sequence length must equal cache chunk_size={kv_cache.chunk_size}; "
                f"got {x.shape[-2]}"
            )

        # Leading input dimensions collapse into the cache's single batch axis:
        # ``[..., L, Q] -> [B, L, Q]`` where ``B = prod(x.shape[:-2])``.
        batch_size = math.prod(x.shape[:-2])
        expected_cache_shape = (
            batch_size,
            self.attention_config.n_heads,
            self.attention_config.head_dim,
        )
        cache_shape = (
            kv_cache._k.shape[0],
            kv_cache._k.shape[2],
            kv_cache._k.shape[3],
        )
        if cache_shape != expected_cache_shape:
            raise ValueError(
                "cache batch, head, and feature dimensions must equal "
                f"{expected_cache_shape}; got {cache_shape}"
            )

        # K/V share shape, device, dtype, and dense layout so the standard
        # cache update preserves the layout required by the attention backend.
        if kv_cache._v.shape != kv_cache._k.shape:
            raise ValueError("Optimized attention requires identical K/V cache shapes")
        if kv_cache._k.device != x.device or kv_cache._v.device != x.device:
            raise RuntimeError("K/V cache tensors must match the input device")
        expected_cache_dtype = (
            torch.float8_e4m3fn
            if self.optimized_impl_config.quantization.quantized_sdpa
            else x.dtype
        )
        if (
            kv_cache._k.dtype != expected_cache_dtype
            or kv_cache._v.dtype != expected_cache_dtype
        ):
            raise RuntimeError(f"K/V cache tensors must use {expected_cache_dtype}")
        # ``is_contiguous`` identifies physical BSHD storage. Transposing S/H and
        # checking again identifies the alternate physical BHSD storage while the
        # tensors retain logical ``[B, S, H, D]`` shapes. Only HEAD normalization
        # supports BHSD because each head's ``[S, D]`` plane must be dense; INNER
        # normalization instead needs each token's complete ``H * D`` row dense.
        token_major = kv_cache._k.is_contiguous() and kv_cache._v.is_contiguous()
        head_major = (
            self.attention_config.qk_norm_scope is QKNormScope.HEAD
            and kv_cache._k.transpose(1, 2).is_contiguous()
            and kv_cache._v.transpose(1, 2).is_contiguous()
        )
        if not token_major and not head_major:
            raise RuntimeError(
                "K/V cache storage must be dense token-major, or head-major for "
                "head-scoped RMSNorm"
            )

        if self.attention_config.rope_config is not None and rope_freqs is not None:
            # RoPE coefficients cover this ``L``-token chunk and broadcast across
            # flattened batches and ``H`` heads inside the shared kernel.
            expected_rope_shape = (x.shape[-2], 1, 1, self.attention_config.head_dim)
            if tuple(rope_freqs.shape) != expected_rope_shape:
                raise ValueError(
                    f"rope_freqs must have shape {expected_rope_shape}; "
                    f"got {tuple(rope_freqs.shape)}"
                )
            if rope_freqs.device != x.device:
                raise RuntimeError("rope_freqs and x must be on the same device")

    # ------------------------ Projection ------------------------ #

    def _prequantize_projection_input(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize one source tensor for reuse across separate projections.

        Args:
            x: Full-precision projection input.

        Returns:
            Quantized activations and their per-slice scale.
        """
        projection_dtype = self.optimized_impl_config.quantization.projection
        assert projection_dtype is not None
        return quantize(x, projection_dtype, Granularity.SLICE, axis=-1)

    def _apply_qk_norm(self, x: Tensor, norm: nn.Module) -> Tensor:
        """Apply PyTorch Q/K normalization with the configured feature scope.

        Args:
            x: Projected query or key shaped ``[B, L, H, D]``.
            norm: PyTorch normalization module or identity.

        Returns:
            Normalized tensor with the same shape as ``x``.
        """
        if self.attention_config.qk_norm_scope is QKNormScope.INNER:
            return norm(x.flatten(-2)).reshape(x.shape)
        return norm(x)

    def _project_query(
        self,
        query: Tensor,
        prequantized_input: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Project and normalize queries in token-major head layout."""
        self._validate_tokens(query, self.attention_config.query_dim, "query")
        sequence_length = query.shape[-2]
        if self.optimized_impl_config.quantization.projection is None:
            projected_query = self.query_projection(query)
        else:
            if self.quantized_query_projection is None:
                raise RuntimeError("quantized query projection is not initialized")
            if prequantized_input is None:
                prequantized_input = self._prequantize_projection_input(query)
            quantized_query, query_scale = prequantized_input
            projected_query = self.quantized_query_projection(
                quantized_query,
                query_scale,
                out_dtype=query.dtype,
            )
        projected_query = projected_query.reshape(
            -1,
            sequence_length,
            self.attention_config.n_heads,
            self.attention_config.head_dim,
        )
        return self._apply_qk_norm(projected_query, self.query_norm)

    def _project_kv(
        self,
        context: Tensor,
        prequantized_input: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Project and normalize context keys in token-major head layout."""
        assert self.attention_config.context_dim is not None
        self._validate_tokens(context, self.attention_config.context_dim, "context")
        sequence_length = context.shape[-2]
        head_shape = (
            -1,
            sequence_length,
            self.attention_config.n_heads,
            self.attention_config.head_dim,
        )
        if self.qkv_fusion_option is QKVFusionOption.NONE:
            if self.optimized_impl_config.quantization.projection is None:
                projected_key = self.key_projection(context)
                projected_value = self.value_projection(context)
            else:
                if (
                    self.quantized_key_projection is None
                    or self.quantized_value_projection is None
                ):
                    raise RuntimeError("quantized K/V projections are not initialized")
                if prequantized_input is None:
                    prequantized_input = self._prequantize_projection_input(context)
                quantized_context, context_scale = prequantized_input
                projected_key = self.quantized_key_projection(
                    quantized_context,
                    context_scale,
                    out_dtype=context.dtype,
                )
                projected_value = self.quantized_value_projection(
                    quantized_context,
                    context_scale,
                    out_dtype=context.dtype,
                )
            key = projected_key.reshape(head_shape)
            value = projected_value.reshape(head_shape)
        else:
            if self.fused_kv is None:
                raise RuntimeError("fused K/V projection is not initialized")
            if self.optimized_impl_config.quantization.projection is None:
                projected_kv = self.fused_kv(context)
            else:
                if not isinstance(self.fused_kv, QuantizedNonPersistentLinear):
                    raise RuntimeError(
                        "quantized fused K/V projection is not initialized"
                    )
                if prequantized_input is None:
                    projected_kv = self.fused_kv(
                        context,
                        Granularity.SLICE,
                        out_dtype=context.dtype,
                    )
                else:
                    quantized_context, context_scale = prequantized_input
                    projected_kv = self.fused_kv(
                        quantized_context,
                        context_scale,
                        out_dtype=context.dtype,
                    )
            projected_kv = projected_kv.reshape(
                -1,
                sequence_length,
                2,
                self.attention_config.n_heads,
                self.attention_config.head_dim,
            )
            key, value = projected_kv.unbind(dim=2)
        return self._apply_qk_norm(key, self.key_norm), value

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V with one fused GEMM."""
        if self.fused_qkv is None:
            raise RuntimeError("fused QKV projection is not initialized")
        if self.optimized_impl_config.quantization.projection is None:
            qkv = self.fused_qkv(x)
        else:
            if not isinstance(self.fused_qkv, QuantizedNonPersistentLinear):
                raise RuntimeError("quantized fused QKV projection is not initialized")
            qkv = self.fused_qkv(x, Granularity.SLICE, out_dtype=x.dtype)
        qkv = qkv.reshape(
            -1,
            x.shape[-2],
            3,
            self.attention_config.n_heads,
            self.attention_config.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        return query, key, value

    def _project_output(self, x: Tensor) -> Tensor:
        """Apply the FP16/BF16 output projection."""
        return self.output_projection(x)


__all__ = [
    "QKVFusionOption",
    "QuantizationOption",
    "SDPABackend",
    "OptimizedImplConfig",
    "OptimizedMultiHeadAttention",
    "flash_attention_2",
    "flash_attention_2_tma",
    "is_tma_flash_attention_supported",
]
