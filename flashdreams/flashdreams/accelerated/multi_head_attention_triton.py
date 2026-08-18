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

"""Triton-accelerated inference-only multi-head attention."""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flashdreams.accelerated.fp8_quantization import fp8_linear, quantize_fp8_weight
from flashdreams.accelerated.multi_head_attention import (
    AttentionType,
    MultiHeadAttention,
    QKNormScope,
)
from flashdreams.accelerated.triton import (
    flash_attention_2_tma,
    fused_rms_rope_kv_cache_update,
)
from flashdreams.core.attention import BlockKVCache


class SDPABackend(str, Enum):
    """Scaled-dot-product attention implementation."""

    CUDNN = "cudnn"
    """Use PyTorch SDPA forced to the cuDNN backend."""

    TRITON = "triton"
    """Use Triton FlashAttention2 (FA2)."""


class QKVFusionOption(str, Enum):
    """Projection fusion policy."""

    NONE = "none"
    """Project queries, keys, and values independently."""

    FULL = "full"
    """Use one QKV GEMM; query and context feature widths must match."""

    FUSE_KV = "fuse_kv"
    """Use an independent Q GEMM and one KV GEMM, allowing unequal input widths."""


class _DerivedProjectionWeights(nn.Module):
    """Hold execution-ready projection tensors derived from master parameters.

    Registering these tensors as nonpersistent buffers gives PyTorch device and
    distributed-module traversal without duplicating them in checkpoints. The
    owning attention module rebuilds them after state loading and device/dtype
    conversion, which also preserves E4M3 storage instead of converting an old
    quantized copy to the requested parameter dtype.
    """

    fused_qkv_weight: Tensor | None
    """Full-fusion QKV weight, shape ``[3 * H * D, Q]``."""

    fused_qkv_bias: Tensor | None
    """Full-fusion QKV bias, shape ``[3 * H * D]``."""

    fused_qkv_weight_scale: Tensor | None
    """Full-fusion FP32 scales, shape ``[3 * H * D]``."""

    fused_kv_weight: Tensor | None
    """Fused K/V weight, shape ``[2 * H * D, C]``; a QKV-tail view for ``FULL``."""

    fused_kv_bias: Tensor | None
    """Fused K/V bias, shape ``[2 * H * D]``; a QKV-tail view for ``FULL``."""

    fused_kv_weight_scale: Tensor | None
    """Fused K/V FP32 scales, shape ``[2 * H * D]``; a QKV-tail view for ``FULL``."""

    q_weight_fp8: Tensor | None
    """E4M3 query-projection weight."""

    q_weight_scale: Tensor | None
    """FP32 query weight scales."""

    k_weight_fp8: Tensor | None
    """E4M3 key-projection weight."""

    k_weight_scale: Tensor | None
    """FP32 key weight scales."""

    v_weight_fp8: Tensor | None
    """E4M3 value-projection weight."""

    v_weight_scale: Tensor | None
    """FP32 value weight scales."""

    output_weight_fp8: Tensor | None
    """E4M3 output-projection weight, shape ``[Q, H * D]``."""

    output_weight_scale: Tensor | None
    """FP32 output weight scales, shape ``[Q]``."""

    def __init__(self) -> None:
        """Reserve every derived tensor as a nonpersistent module buffer."""
        super().__init__()
        # Registering ``None`` reserves each name in ``Module._buffers``. Later
        # tensor assignments therefore remain visible to ``Module.to`` and
        # distributed wrappers while ``persistent=False`` excludes them from the
        # Torch-compatible state dict.
        self.register_buffer("fused_qkv_weight", None, persistent=False)
        self.register_buffer("fused_qkv_bias", None, persistent=False)
        self.register_buffer("fused_qkv_weight_scale", None, persistent=False)
        self.register_buffer("fused_kv_weight", None, persistent=False)
        self.register_buffer("fused_kv_bias", None, persistent=False)
        self.register_buffer("fused_kv_weight_scale", None, persistent=False)
        self.register_buffer("q_weight_fp8", None, persistent=False)
        self.register_buffer("q_weight_scale", None, persistent=False)
        self.register_buffer("k_weight_fp8", None, persistent=False)
        self.register_buffer("k_weight_scale", None, persistent=False)
        self.register_buffer("v_weight_fp8", None, persistent=False)
        self.register_buffer("v_weight_scale", None, persistent=False)
        self.register_buffer("output_weight_fp8", None, persistent=False)
        self.register_buffer("output_weight_scale", None, persistent=False)


def _cache_write_slice(kv_cache: BlockKVCache) -> tuple[int, int, int]:
    """Map the current token chunk onto its physical cache write interval.

    The cache normally consumes the complete ``[B, L, H, D]`` K/V chunk. When a
    rolling write would overlap an immutable sink prefix, only the trailing
    source tokens that fit after the sink are copied.

    Args:
        kv_cache: Block cache prepared for the current chunk.

    Returns:
        Source-token offset, destination-cache offset, and token count for the
        fused K/V write.
    """
    write_start, write_end = kv_cache._current_write_bounds()
    read_start = 0
    if (
        kv_cache.sink_size > 0
        and not kv_cache._current_chunk_overlaps_sink()
        and write_start < kv_cache.sink_size
    ):
        # Preserve the sink and align the source tail with the write interval.
        # This mirrors BlockKVCache.update without staging processed K/V.
        write_start = kv_cache.sink_size
        write_length = write_end - write_start
        read_start = kv_cache.chunk_size - write_length
    return int(read_start), int(write_start), int(write_end - write_start)


class TritonMultiHeadAttention(MultiHeadAttention[BlockKVCache]):
    """Provide inference-only streaming self- and static cross-attention.

    Shape comments use ``B`` for the product of all leading batch dimensions,
    ``L`` for the current query/chunk length, ``S`` for visible cached context,
    ``H`` for the number of heads, ``D`` for the head dimension, ``Q`` for
    ``query_dim``, and ``C`` for ``context_dim``.

    Full-fusion self-attention produces Q/K/V in one GEMM and keeps processed Q
    local to :meth:`forward` while a Triton kernel writes normalized, rotated K/V
    directly into cache storage. Cross-attention precomputes K/V independently
    and reuses that static cache across forward calls.

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

    use_fp8: bool
    """Whether projection/output GEMMs and supported attention storage use FP8."""

    sdpa_backend: SDPABackend
    """Scaled-dot-product attention implementation selected at construction."""

    qkv_fusion_option: QKVFusionOption
    """Projection fusion policy selected at construction."""

    _derived_weights: _DerivedProjectionWeights
    """Nonpersistent native and FP8 tensors derived from projection parameters."""

    def __init__(
        self,
        query_dim: int,
        n_heads: int = 8,
        head_dim: int = 64,
        *,
        context_dim: int | None = None,
        attention_type: AttentionType = AttentionType.SELF_ATTENTION,
        qkv_fusion_option: QKVFusionOption = QKVFusionOption.FULL,
        qk_norm_eps: float = 1e-6,
        qk_norm_scope: QKNormScope = QKNormScope.HEAD,
        rope_interleaved: bool = False,
        use_fp8: bool = False,
        sdpa_backend: SDPABackend = SDPABackend.CUDNN,
    ) -> None:
        """Initialize shared Triton attention geometry and policies.

        Args:
            query_dim: Feature dimension of input and output tokens.
            n_heads: Number of query, key, and value heads.
            head_dim: Feature dimension of each attention head.
            context_dim: Feature dimension projected into keys and values.
                Defaults to ``query_dim``.
            attention_type: Whether forward performs self- or cross-attention.
            qkv_fusion_option: Projection fusion policy. Full QKV fusion requires
                equal query and context feature dimensions.
            qk_norm_eps: Epsilon used by Q/K RMS normalization.
            qk_norm_scope: Feature scope used by Q/K RMS normalization, or
                :attr:`QKNormScope.NONE` to disable normalization.
            rope_interleaved: Rotate adjacent feature pairs instead of half splits.
            use_fp8: Use row-scaled E4M3 projection/output GEMMs and E4M3
                attention/cache storage with the Triton backend. cuDNN uses the
                native activation dtype for attention and cache storage.
            sdpa_backend: Scaled-dot-product attention implementation.

        Raises:
            TypeError: A fusion or SDPA backend policy has the wrong type.
            ValueError: Full QKV fusion has mismatched input widths,
                ``head_dim`` is unsupported, or an FP8 input width is not
                aligned to 16 features.
        """
        super().__init__(
            query_dim=query_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            context_dim=context_dim,
            attention_type=attention_type,
            qk_norm_eps=qk_norm_eps,
            qk_norm_scope=qk_norm_scope,
            rope_interleaved=rope_interleaved,
        )
        if not isinstance(qkv_fusion_option, QKVFusionOption):
            raise TypeError(
                "qkv_fusion_option must be a QKVFusionOption; "
                f"got {qkv_fusion_option!r}"
            )
        if (
            qkv_fusion_option is QKVFusionOption.FULL
            and self.query_dim != self.context_dim
        ):
            raise ValueError(
                "full QKV fusion requires query_dim to equal context_dim; "
                f"got {self.query_dim} and {self.context_dim}"
            )
        if not isinstance(sdpa_backend, SDPABackend):
            raise TypeError(
                f"sdpa_backend must be an SDPABackend; got {sdpa_backend!r}"
            )
        if not (16 <= head_dim <= 256 and head_dim & (head_dim - 1) == 0):
            raise ValueError(
                "accelerated attention requires a power-of-two head_dim in [16, 256]; "
                f"got {head_dim}"
            )
        if use_fp8 and (query_dim % 16 != 0 or self.context_dim % 16 != 0):
            raise ValueError(
                "FP8 projections require query_dim and context_dim to be multiples "
                f"of 16; got {query_dim} and {self.context_dim}"
            )

        self.qkv_fusion_option = qkv_fusion_option
        self.use_fp8 = use_fp8
        self.sdpa_backend = sdpa_backend

    # ---------------------- Initialization ---------------------- #

    def _initialize_derived_weights(self) -> None:
        """Build execution weights after concrete checkpoint fields exist.

        Concrete implementations call this after assigning their checkpoint
        fields. The logical accessors are valid before fusion buffers or load
        hooks read any projection or normalization module.
        """
        self._derived_weights = _DerivedProjectionWeights()
        self._refresh_derived_weights()
        self.register_load_state_dict_post_hook(self._refresh_derived_weights)

    @torch.no_grad()
    def _refresh_derived_weights(self, *args: object) -> None:
        """Rebuild execution weights from the registered projection parameters.

        The logical projection accessors remain the source of truth even when a
        subclass registers them under model-specific names.
        This method materializes only the fusion/precision representation selected
        for inference, and runs after strict state loading as well as module moves.

        Args:
            args: ``(module, incompatible_keys)`` supplied by PyTorch when invoked
                as a load-state-dict post-hook; ignored on both hook and direct calls.
        """
        del args

        # Select non-overlapping source matrices so each active group is
        # quantized exactly once. FULL derives every projection from QKV;
        # FUSE_KV keeps Q separate; NONE keeps Q, K, and V separate.
        qkv_weight: Tensor | None = None
        qkv_bias: Tensor | None = None
        kv_weight: Tensor | None = None
        kv_bias: Tensor | None = None
        q_weight: Tensor | None = self.query_projection.weight
        k_weight: Tensor | None = self.key_projection.weight
        v_weight: Tensor | None = self.value_projection.weight
        if self.qkv_fusion_option is QKVFusionOption.FULL:
            qkv_weight = torch.cat(
                (
                    self.query_projection.weight,
                    self.key_projection.weight,
                    self.value_projection.weight,
                ),
                dim=0,
            ).detach()
            q_weight = k_weight = v_weight = None
            if self.query_projection.bias is not None:
                assert (
                    self.key_projection.bias is not None
                    and self.value_projection.bias is not None
                )
                qkv_bias = torch.cat(
                    (
                        self.query_projection.bias,
                        self.key_projection.bias,
                        self.value_projection.bias,
                    ),
                    dim=0,
                ).detach()
        elif self.qkv_fusion_option is QKVFusionOption.FUSE_KV:
            kv_weight = torch.cat(
                (self.key_projection.weight, self.value_projection.weight), dim=0
            ).detach()
            k_weight = v_weight = None
            if self.key_projection.bias is not None:
                assert self.value_projection.bias is not None
                kv_bias = torch.cat(
                    (self.key_projection.bias, self.value_projection.bias), dim=0
                ).detach()

        # Clear every execution buffer before repopulating the active policy.
        # Refreshes after state loading or module moves cannot retain stale views.
        derived = self._derived_weights
        derived.fused_qkv_weight = None
        derived.fused_qkv_bias = None
        derived.fused_qkv_weight_scale = None
        derived.fused_kv_weight = None
        derived.fused_kv_bias = None
        derived.fused_kv_weight_scale = None
        derived.q_weight_fp8 = None
        derived.q_weight_scale = None
        derived.k_weight_fp8 = None
        derived.k_weight_scale = None
        derived.v_weight_fp8 = None
        derived.v_weight_scale = None
        derived.output_weight_fp8 = None
        derived.output_weight_scale = None

        if qkv_bias is not None:
            derived.fused_qkv_bias = qkv_bias
            derived.fused_kv_bias = qkv_bias[self.inner_dim :]
        if kv_bias is not None:
            derived.fused_kv_bias = kv_bias

        if not self.use_fp8:
            if qkv_weight is not None:
                native_qkv_weight = qkv_weight.contiguous()
                derived.fused_qkv_weight = native_qkv_weight
                derived.fused_kv_weight = native_qkv_weight[self.inner_dim :]
            if kv_weight is not None:
                derived.fused_kv_weight = kv_weight.contiguous()
            return

        if qkv_weight is not None:
            # Q/K/V and fused K/V share one E4M3 weight and scale allocation.
            qkv_fp8, qkv_scale = quantize_fp8_weight(qkv_weight)
            derived.fused_qkv_weight = qkv_fp8
            derived.fused_qkv_weight_scale = qkv_scale
            derived.fused_kv_weight = qkv_fp8[self.inner_dim :]
            derived.fused_kv_weight_scale = qkv_scale[self.inner_dim :]
            (
                derived.q_weight_fp8,
                derived.k_weight_fp8,
                derived.v_weight_fp8,
            ) = qkv_fp8.split(self.inner_dim)
            (
                derived.q_weight_scale,
                derived.k_weight_scale,
                derived.v_weight_scale,
            ) = qkv_scale.split(self.inner_dim)

        if kv_weight is not None:
            # K/V share the fused allocation while Q remains independent.
            kv_fp8, kv_scale = quantize_fp8_weight(kv_weight)
            derived.fused_kv_weight = kv_fp8
            derived.fused_kv_weight_scale = kv_scale
            derived.k_weight_fp8, derived.v_weight_fp8 = kv_fp8.split(self.inner_dim)
            derived.k_weight_scale, derived.v_weight_scale = kv_scale.split(
                self.inner_dim
            )

        if q_weight is not None:
            derived.q_weight_fp8, derived.q_weight_scale = quantize_fp8_weight(q_weight)
        if k_weight is not None:
            derived.k_weight_fp8, derived.k_weight_scale = quantize_fp8_weight(k_weight)
        if v_weight is not None:
            derived.v_weight_fp8, derived.v_weight_scale = quantize_fp8_weight(v_weight)

        # Output consumes attention results and always owns its quantized matrix.
        (
            derived.output_weight_fp8,
            derived.output_weight_scale,
        ) = quantize_fp8_weight(self.output_projection.weight)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> TritonMultiHeadAttention:
        """Transform parameters and rebuild derived weights on their final device.

        Args:
            fn: Tensor transformation applied by :class:`torch.nn.Module`.
            recurse: Apply ``fn`` recursively to child modules.

        Returns:
            This module with derived projection buffers refreshed.
        """
        # Move/cast canonical parameters first, then regenerate from those final
        # values. Applying ``fn`` directly to an existing E4M3 buffer would either
        # change its dtype or preserve scales computed for stale master weights.
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
        """Allocate a rolling cache matching the configured precision policy.

        Args:
            batch_size: Flattened batch size ``B``.
            chunk_size: Number of current tokens ``L`` written per update.
            window_size: Number of rolling context tokens retained after the sink.
            sink_size: Number of initial context tokens that are never evicted.
            device: Device on which to allocate K/V storage.
            dtype: Native activation dtype. The Triton FA2 backend uses E4M3 cache
                storage when FP8 is enabled; cuDNN retains this dtype.

        Returns:
            Block cache with K/V storage shaped
            ``[B, sink_size + window_size, H, D]``.

        Raises:
            TypeError: FP8 is enabled with an activation dtype other than FP16 or
                BF16.
        """
        # Keep ``[B, S, H, D]`` as the public cache shape: ``BlockKVCache``
        # rolls and slices axis 1, and both attention backends accept that logical
        # order.
        cache_shape = (
            batch_size,
            sink_size + window_size,
            self.n_heads,
            self.head_dim,
        )
        if self.use_fp8 and dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("FP8 projections require FP16 or BF16 activations")
        cache_dtype = (
            torch.float8_e4m3fn
            if self.use_fp8 and self.sdpa_backend is SDPABackend.TRITON
            else dtype
        )
        cache = BlockKVCache(
            k_shape=cache_shape,
            v_shape=cache_shape,
            seq_dim=1,
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            device=device,
            dtype=cache_dtype,
        )
        if self.qk_norm_scope is QKNormScope.HEAD:
            # ``BlockKVCache`` initially allocates a contiguous ``[B, S, H, D]``
            # tensor, whose physical order makes all ``H * D`` features for one
            # token adjacent. Per-head normalization and cuDNN attention instead
            # consume one head's complete ``[S, D]`` plane at a time. Because the
            # allocation is still empty, reinterpret the same bytes as contiguous
            # ``[B, H, S, D]``; ``view`` changes sizes/strides but copies nothing.
            storage_shape = (
                batch_size,
                self.n_heads,
                sink_size + window_size,
                self.head_dim,
            )
            # Transposing the H/S metadata restores the public ``[B, S, H, D]``
            # shape while retaining physical BHSD order. For example, the final
            # strides are ``[H*S*D, D, S*D, 1]``: advancing a token within one
            # head moves by ``D`` elements, so ``cache[:, :, h, :]`` is dense.
            # A later ``transpose(1, 2)`` recovers physical ``[B, H, S, D]`` order
            # for cuDNN without rearranging bytes. The full cache is contiguous;
            # a shorter filling-phase prefix retains valid head-major strides.
            cache._k = cache._k.view(storage_shape).transpose(1, 2)
            cache._v = cache._v.view(storage_shape).transpose(1, 2)
        return cache

    @torch.no_grad()
    def compute_kv(
        self,
        context: Tensor,
        rope_freqs: Tensor | None = None,
    ) -> BlockKVCache:
        """Project complete context into a reusable static K/V cache.

        Args:
            context: Context tokens shaped ``[..., S, C]``.
            rope_freqs: Optional key rotation angles shaped ``[S, 1, 1, D]``.

        Returns:
            Filled cache with logical K/V shape ``[B, S, H, D]``. Its sequence
            length and window both equal S, so subsequent forward calls read all
            context.
        """
        key, value = self._project_kv(context)
        if rope_freqs is not None:
            # Position affects key directions used in Q·K; values are never rotated.
            key = self._apply_rope(key, rope_freqs)

        # Convert only after normalization/RoPE, keeping those numerically sensitive
        # operations in the native activation dtype.
        key = self._attention_storage(key)
        value = self._attention_storage(value)
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
            rope_freqs: Optional query rotation angles shaped ``[L, 1, 1, D]``.
                Self-attention also applies them to current keys.

        Returns:
            Output-projected tokens with the same shape and dtype as ``x``.
        """
        if self.attention_type is AttentionType.SELF_ATTENTION:
            query = self._update_kv_and_compute_query(x, kv_cache, rope_freqs)
        else:
            query = self._compute_query(x, rope_freqs)
            self._validate_cache(kv_cache, x)

        # ``cached_k/v`` expose only the valid prefix while a rolling cache fills,
        # and the complete fixed-size buffer after it reaches steady state.
        output = self._attention(
            query,
            kv_cache.cached_k(),
            kv_cache.cached_v(),
        )
        sequence_length = x.shape[-2]
        output = output.reshape(-1, sequence_length, self.inner_dim)
        output = self._project_output(output, x.dtype)
        return output.reshape(x.shape[:-2] + (sequence_length, self.query_dim))

    # ------------------------------------------------------------ #
    #                        Private Method                        #
    # ------------------------------------------------------------ #

    # ------------------ Core Attention Methods ------------------ #

    def _compute_query(
        self,
        query: Tensor,
        rope_freqs: Tensor | None,
    ) -> Tensor:
        """Project, normalize, and optionally rotate query tokens."""
        query = self._project_query(query)
        if rope_freqs is not None:
            query = self._apply_rope(query, rope_freqs)
        return self._attention_storage(query)

    def _update_kv_and_compute_query(
        self,
        x: Tensor,
        kv_cache: BlockKVCache,
        rope_freqs: Tensor | None,
    ) -> Tensor:
        """Update rolling K/V and return the processed current query."""
        if self.qkv_fusion_option is QKVFusionOption.FULL:
            self._validate_fused_update_inputs(x, kv_cache, rope_freqs)
            sequence_length = x.shape[-2]
            x_flat = x.reshape(-1, sequence_length, self.query_dim)
            query, key, value = self._project_qkv(x_flat)
            (
                cache_read_start,
                cache_write_start,
                cache_write_length,
            ) = _cache_write_slice(kv_cache)

            if self.qk_norm_scope is QKNormScope.NONE:
                if not isinstance(self.query_norm, nn.Identity) or not isinstance(
                    self.key_norm, nn.Identity
                ):
                    raise RuntimeError(
                        "Q/K normalization modules must use the same policy"
                    )
                query_weight: Tensor | None = None
                key_weight: Tensor | None = None
            else:
                if not isinstance(self.query_norm, nn.RMSNorm) or not isinstance(
                    self.key_norm, nn.RMSNorm
                ):
                    raise RuntimeError(
                        "Q/K normalization modules must use the same policy"
                    )
                query_weight = self.query_norm.weight
                key_weight = self.key_norm.weight

            # Keep processed Q local while the fused kernel writes K/V directly
            # into the current physical cache interval.
            return fused_rms_rope_kv_cache_update(
                query,
                key,
                value,
                kv_cache._k,
                kv_cache._v,
                query_weight=query_weight,
                key_weight=key_weight,
                norm_eps=self.qk_norm_eps,
                norm_scope=self.qk_norm_scope,
                rope_freqs=rope_freqs,
                rope_interleaved=self.rope_interleaved,
                cache_read_start=cache_read_start,
                cache_write_start=cache_write_start,
                cache_write_length=cache_write_length,
            )

        self._validate_tokens(x, self.context_dim, "context")
        self._validate_cache(kv_cache, x)
        if kv_cache._curr_chunk_idx is None:
            raise RuntimeError("call kv_cache.before_update() before attention")
        if x.shape[-2] != kv_cache.chunk_size:
            raise ValueError(
                "context sequence length must equal cache "
                f"chunk_size={kv_cache.chunk_size}; got {x.shape[-2]}"
            )

        query = self._compute_query(x, rope_freqs)
        key, value = self._project_kv(x)
        if rope_freqs is not None:
            key = self._apply_rope(key, rope_freqs)
        kv_cache.update(self._attention_storage(key), self._attention_storage(value))
        return query

    def _attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Apply the configured non-causal scaled-dot-product attention backend.

        Args:
            query: Processed queries with shape ``[B, L, H, D]``.
            key: Cached keys with shape ``[B, S, H, D]``.
            value: Cached values with shape ``[B, S, H, D]``.

        Returns:
            Attention output with shape ``[B, L, H, D]``.
        """
        if self.sdpa_backend is SDPABackend.CUDNN:
            # The module and Triton kernel use token-major ``[B, L/S, H, D]``.
            # PyTorch SDPA instead interprets its two middle axes as ``[H, L/S]``.
            # These transposes normally change only shape/stride metadata. A cache
            # allocated in physical BHSD order exposes each head's S/D plane
            # directly here; selecting its full sequence extent is contiguous.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # Force cuDNN rather than allowing PyTorch to silently fall back to a
            # backend with different performance or supported-layout behavior.
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.CUDNN_ATTENTION
            ):
                output = F.scaled_dot_product_attention(query, key, value)

            # Restore the module-wide ``[B, L, H, D]`` contract for head merging.
            return output.transpose(1, 2)

        # The TMA wrapper natively consumes and returns token-major BSHD tensors.
        return flash_attention_2_tma(query, key, value)

    def _apply_rope(self, x: Tensor, rope_freqs: Tensor) -> Tensor:
        """Apply rotary position embeddings to token-major head features.

        Args:
            x: Projected Q or K tensor shaped ``[..., L, H, D]``.
            rope_freqs: Rotation angles shaped ``[L, 1, 1, D]``.

        Returns:
            Rotated tensor with the same shape and dtype as ``x``.

        Raises:
            ValueError: ``D`` is odd or the angle tensor has the wrong shape.
            RuntimeError: Angles and projected tokens occupy different devices.
        """
        if x.shape[-1] % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim; got {x.shape[-1]}")
        expected_shape = (x.shape[-3], 1, 1, x.shape[-1])
        if tuple(rope_freqs.shape) != expected_shape:
            raise ValueError(
                f"rope_freqs must have shape {expected_shape}; "
                f"got {tuple(rope_freqs.shape)}"
            )
        if rope_freqs.device != x.device:
            raise RuntimeError("rope_freqs and tokens must be on the same device")
        # Remove the two singleton axes supplied by recipe code, then prepend
        # enough singleton batch axes to broadcast ``[L, 1, D]`` over every
        # leading batch and all H heads without materializing repeated angles.
        freqs = rope_freqs[:, 0, 0, :].reshape(
            (1,) * (x.ndim - 3) + (x.shape[-3], 1, x.shape[-1])
        )
        cos_freqs = torch.cos(freqs).to(dtype=x.dtype)
        sin_freqs = torch.sin(freqs).to(dtype=x.dtype)

        # Build R(x), the vector rotated 90 degrees inside every feature pair.
        # Interleaved RoPE pairs (0,1), (2,3), ...; split-half RoPE pairs feature
        # i with i + D/2. ``x*cos(theta) + R(x)*sin(theta)`` then performs all
        # independent 2-D rotations in parallel.
        if self.rope_interleaved:
            rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
        else:
            first, second = x.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
        return x * cos_freqs + rotated * sin_freqs

    def _attention_storage(self, x: Tensor) -> Tensor:
        """Convert processed Q/K/V to the configured attention storage dtype.

        Args:
            x: Native FP16/BF16 projected attention tensor.

        Returns:
            E4M3 storage for FP8 Triton FA2, otherwise ``x`` unchanged.

        PyTorch's cuDNN SDPA does not accept FP8 Q/K/V, so ``use_fp8`` affects
        its projection GEMMs but leaves attention and cache storage native.
        """
        if self.use_fp8 and self.sdpa_backend is SDPABackend.TRITON:
            return x.to(torch.float8_e4m3fn)
        return x

    # ------------------------ Validation ------------------------ #

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
                "TritonMultiHeadAttention requires CUDA FP16 or BF16 inputs"
            )
        if torch.cuda.get_device_capability(x.device)[0] < 9:
            raise RuntimeError(
                "TritonMultiHeadAttention requires compute capability 9.0 or newer"
            )

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
                "TritonMultiHeadAttention requires a [B, S, H, D] cache with seq_dim=1"
            )
        # Public leading dimensions such as batch and video view collapse into
        # one B axis before projection; cached K/V must use the same flattening.
        expected_shape = (math.prod(x.shape[:-2]), self.n_heads, self.head_dim)
        cache_shape = (kv_cache._k.shape[0], kv_cache._k.shape[2], kv_cache._k.shape[3])
        if cache_shape != expected_shape:
            raise ValueError(
                "cache batch, head, and feature dimensions must equal "
                f"{expected_shape}; got {cache_shape}"
            )
        if kv_cache._v.shape != kv_cache._k.shape:
            raise ValueError("Triton attention requires identical K/V cache shapes")
        if kv_cache._k.device != x.device or kv_cache._v.device != x.device:
            raise RuntimeError("K/V cache tensors must match the input device")
        expected_dtype = (
            torch.float8_e4m3fn
            if self.use_fp8 and self.sdpa_backend is SDPABackend.TRITON
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
        if x.shape[-1] != self.query_dim:
            raise ValueError(
                f"x feature width must equal query_dim={self.query_dim}; "
                f"got {x.shape[-1]}"
            )

        # The accelerated path accepts native FP16/BF16 CUDA inputs; FP8 is
        # an internal projection and FA2 attention/cache-storage policy.
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "TritonMultiHeadAttention requires CUDA FP16 or BF16 inputs"
            )
        if torch.cuda.get_device_capability(x.device)[0] < 9:
            raise RuntimeError(
                "TritonMultiHeadAttention requires compute capability 9.0 or newer"
            )

        # The caller prepares cache write bounds before attention. K/V storage
        # keeps logical ``[B, S, H, D]`` axes in one of two supported dense layouts.
        if kv_cache._curr_chunk_idx is None:
            raise RuntimeError("call kv_cache.before_update() before attention")
        if kv_cache.seq_dim != 1 or kv_cache._k.ndim != 4:
            raise ValueError(
                "TritonMultiHeadAttention requires a [B, S, H, D] cache with seq_dim=1"
            )
        if x.shape[-2] != kv_cache.chunk_size:
            raise ValueError(
                f"x sequence length must equal cache chunk_size={kv_cache.chunk_size}; "
                f"got {x.shape[-2]}"
            )

        # Leading input dimensions collapse into the cache's single batch axis:
        # ``[..., L, Q] -> [B, L, Q]`` where ``B = prod(x.shape[:-2])``.
        batch_size = math.prod(x.shape[:-2])
        expected_cache_shape = (batch_size, self.n_heads, self.head_dim)
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

        # K/V share shape, device, storage precision, and dense layout so one
        # fused kernel can write them for the configured attention backend.
        if kv_cache._v.shape != kv_cache._k.shape:
            raise ValueError("Triton attention requires identical K/V cache shapes")
        if kv_cache._k.device != x.device or kv_cache._v.device != x.device:
            raise RuntimeError("K/V cache tensors must match the input device")
        expected_cache_dtype = (
            torch.float8_e4m3fn
            if self.use_fp8 and self.sdpa_backend is SDPABackend.TRITON
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
            self.qk_norm_scope is QKNormScope.HEAD
            and kv_cache._k.transpose(1, 2).is_contiguous()
            and kv_cache._v.transpose(1, 2).is_contiguous()
        )
        if not token_major and not head_major:
            raise RuntimeError(
                "K/V cache storage must be dense token-major, or head-major for "
                "head-scoped RMSNorm"
            )

        if rope_freqs is not None:
            # RoPE coefficients cover this ``L``-token chunk and broadcast across
            # flattened batches and ``H`` heads inside the fused kernel.
            expected_rope_shape = (x.shape[-2], 1, 1, self.head_dim)
            if tuple(rope_freqs.shape) != expected_rope_shape:
                raise ValueError(
                    f"rope_freqs must have shape {expected_rope_shape}; "
                    f"got {tuple(rope_freqs.shape)}"
                )
            if rope_freqs.device != x.device:
                raise RuntimeError("rope_freqs and x must be on the same device")

    # ------------------------ Projection ------------------------ #

    def _project_linear(
        self,
        x: Tensor,
        layer: nn.Linear,
        weight_fp8: Tensor | None,
        weight_scale: Tensor | None,
    ) -> Tensor:
        """Apply one native or row-scaled FP8 projection.

        Args:
            x: Tokens shaped ``[..., length, layer.in_features]``.
            layer: Canonical projection supplying native parameters and bias.
            weight_fp8: E4M3 execution weight shaped like ``layer.weight``.
            weight_scale: Per-output-row dequantization scales.

        Returns:
            Projected tokens in ``x.dtype`` with final width
            ``layer.out_features``.

        Raises:
            RuntimeError: FP8 execution is selected without a derived weight/scale.
        """
        if not self.use_fp8:
            return layer(x)
        if weight_fp8 is None or weight_scale is None:
            raise RuntimeError("FP8 projection weight is not initialized")
        return fp8_linear(x, weight_fp8, weight_scale, layer.bias, x.dtype)

    def _project_query(self, query: Tensor) -> Tensor:
        """Project and normalize queries in token-major head layout.

        Args:
            query: Query tokens shaped ``[..., L, Q]``.

        Returns:
            Flattened-batch queries shaped ``[B, L, H, D]``.
        """
        self._validate_tokens(query, self.query_dim, "query")
        sequence_length = query.shape[-2]

        # The projection emits one ``H * D`` feature vector per token. Splitting
        # that final axis exposes heads while ``-1`` folds every public leading
        # batch dimension into the kernel's single B axis.
        query = self._project_linear(
            query,
            self.query_projection,
            self._derived_weights.q_weight_fp8,
            self._derived_weights.q_weight_scale,
        ).reshape(-1, sequence_length, self.n_heads, self.head_dim)

        if self.qk_norm_scope is QKNormScope.INNER:
            # INNER computes one RMS over all ``H * D`` features of a token.
            # Flattening only the last two axes preserves B/L; reshape restores
            # the head axis expected by attention.
            query = self.query_norm(query.flatten(-2)).reshape(query.shape)
        else:
            # HEAD leaves D last, so RMSNorm runs independently for each head.
            # With NONE, ``q_norm`` is Identity and the same layout passes through.
            query = self.query_norm(query)
        return query

    def _project_kv(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Project and normalize context keys in token-major head layout.

        Args:
            context: Context tokens shaped ``[..., S, C]``.

        Returns:
            Flattened-batch key and value tensors shaped ``[B, S, H, D]``.

        Raises:
            RuntimeError: The selected fused/FP8 execution weights are unavailable.
        """
        self._validate_tokens(context, self.context_dim, "context")
        sequence_length = context.shape[-2]
        head_shape = (-1, sequence_length, self.n_heads, self.head_dim)
        if self.qkv_fusion_option is QKVFusionOption.NONE:
            # Independent K and V matrices each produce ``H * D`` features.
            key = self._project_linear(
                context,
                self.key_projection,
                self._derived_weights.k_weight_fp8,
                self._derived_weights.k_weight_scale,
            ).reshape(head_shape)
            value = self._project_linear(
                context,
                self.value_projection,
                self._derived_weights.v_weight_fp8,
                self._derived_weights.v_weight_scale,
            ).reshape(head_shape)
        else:
            # Both fused policies expose exactly ``[K rows; V rows]`` here. FULL
            # stores a view of its QKV tail; FUSE_KV owns the K/V allocation.
            fused_weight = self._derived_weights.fused_kv_weight
            fused_bias = self._derived_weights.fused_kv_bias
            fused_scale = self._derived_weights.fused_kv_weight_scale

            if fused_weight is None:
                raise RuntimeError("fused K/V weight is not initialized")
            if self.use_fp8:
                if fused_scale is None:
                    raise RuntimeError("FP8 K/V weight scales are not initialized")
                projected_kv = fp8_linear(
                    context,
                    fused_weight,
                    fused_scale,
                    fused_bias,
                    context.dtype,
                )
            else:
                projected_kv = F.linear(context, fused_weight, fused_bias)
            # The fused output axis is ``[K(H*D), V(H*D)]``. Expose that leading
            # K/V selector before H and D, then remove it with zero-copy views.
            projected_kv = projected_kv.reshape(
                -1,
                sequence_length,
                2,
                self.n_heads,
                self.head_dim,
            )
            key, value = projected_kv.unbind(dim=2)

        # Normalize keys because their scale affects Q·K logits; values carry
        # payload features and intentionally bypass Q/K normalization.
        if self.qk_norm_scope is QKNormScope.INNER:
            key = self.key_norm(key.flatten(-2)).reshape(key.shape)
        else:
            key = self.key_norm(key)
        return key, value

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project Q/K/V with one fused native or row-scaled FP8 GEMM.

        Args:
            x: Flattened-batch input tokens, shape ``[B, L, Q]``.

        Returns:
            Query, key, and value tensors, each shaped ``[B, L, H, D]``.

        Raises:
            RuntimeError: A required derived QKV weight or scale is unavailable.
        """
        if self._derived_weights.fused_qkv_weight is None:
            raise RuntimeError("fused QKV weight is not initialized")
        if self.use_fp8:
            if self._derived_weights.fused_qkv_weight_scale is None:
                raise RuntimeError("FP8 QKV weight scales are not initialized")
            # Quantize ``B * L`` activation rows independently, then multiply
            # ``[B * L, Q] @ [Q, 3 * H * D]`` and restore ``x.dtype``.
            qkv = fp8_linear(
                x,
                self._derived_weights.fused_qkv_weight,
                self._derived_weights.fused_qkv_weight_scale,
                self._derived_weights.fused_qkv_bias,
                x.dtype,
            )
        else:
            # ``[B, L, Q] @ [Q, 3 * H * D] -> [B, L, 3 * H * D]``.
            qkv = F.linear(
                x,
                self._derived_weights.fused_qkv_weight,
                self._derived_weights.fused_qkv_bias,
            )

        # Split the fused projection axis into Q/K/V, heads, and head features:
        # ``[B, L, 3 * H * D] -> [B, L, 3, H, D]``.
        qkv = qkv.reshape(
            -1,
            x.shape[-2],
            3,
            self.n_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        return query, key, value

    def _project_output(self, x: Tensor, output_dtype: torch.dtype) -> Tensor:
        """Apply the native or row-scaled FP8 output projection.

        Args:
            x: Head-concatenated attention output, shape ``[B, L, H * D]``.
            output_dtype: Native activation dtype returned to the caller.

        Returns:
            Projected tokens with shape ``[B, L, Q]``.

        Raises:
            RuntimeError: FP8 is enabled but its derived output weight is missing.
        """
        if not self.use_fp8:
            # ``[B, L, H * D] @ [H * D, Q] -> [B, L, Q]``.
            return self.output_projection(x)
        if (
            self._derived_weights.output_weight_fp8 is None
            or self._derived_weights.output_weight_scale is None
        ):
            raise RuntimeError("FP8 output weight is not initialized")
        # Quantize ``B * L`` attention rows independently before the scaled GEMM.
        return fp8_linear(
            x,
            self._derived_weights.output_weight_fp8,
            self._derived_weights.output_weight_scale,
            self.output_projection.bias,
            output_dtype,
        )


__all__ = ["QKVFusionOption", "SDPABackend", "TritonMultiHeadAttention"]
