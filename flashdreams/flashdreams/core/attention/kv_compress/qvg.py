# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quant-VideoGen-style KV storage backend.

The default ``kernel_impl="official_triton"`` path uses official
Quant-VideoGen Triton kernels vendored under ``qvg_triton``. This keeps
FlashDreams independent from the external QVG Python package while preserving an
alignment path against the upstream kernel implementation.

The optional ``kernel_impl="native"`` path is a PyTorch implementation of the
QVG storage idea for tests, debugging, and fallback comparisons.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from flashdreams.core.attention.kv_compress.base import (
    KVCompressionConfig,
    KVSpan,
    KVStorageBackend,
    KVStoragePayload,
    RuntimePhase,
    estimate_tensor_tree_bytes,
)


@dataclass
class QVGQuantConfig:
    """Configuration for QVG PRQ KV compression."""

    quant_type: str = "triton-nstages-kmeans-int2"
    cache_num_k_centroids: int = 256
    cache_num_v_centroids: int = 256
    kmeans_max_iters: int = 2
    quant_block_size: int = 64
    num_prq_stages: int = 1
    scale_dtype: str = "float8_e4m3fn"
    kmeans_init: str = "random"
    kmeans_seed: int | None = None
    cache_k_num_bits: int | None = None
    cache_v_num_bits: int | None = None
    kernel_impl: str = "official_triton"
    preserve_rng: bool = True
    store_prerope_keys: bool = False

    @classmethod
    def from_config(cls, config: KVCompressionConfig) -> "QVGQuantConfig":
        """Create QVG config from a generic compression config."""
        values = dict(config.backend_config)
        return cls(**values)

    def _num_bits_from_quant_type(self) -> int:
        """Extract integer bit width from ``quant_type``."""
        match = re.search(r"int(\d+)", self.quant_type)
        if match is None:
            raise ValueError(f"Cannot infer bit width from {self.quant_type!r}")
        num_bits = int(match.group(1))
        self._validate_num_bits(num_bits)
        return num_bits

    @staticmethod
    def _validate_num_bits(num_bits: int) -> None:
        if num_bits not in (2, 4):
            raise ValueError(f"QVG backend currently supports INT2/INT4, got {num_bits}")

    @property
    def num_bits(self) -> int:
        """Default integer bit width from ``quant_type``."""
        return self._num_bits_from_quant_type()

    @property
    def k_num_bits(self) -> int:
        """Integer bit width for K residuals."""
        num_bits = self.cache_k_num_bits or self.num_bits
        self._validate_num_bits(num_bits)
        return num_bits

    @property
    def v_num_bits(self) -> int:
        """Integer bit width for V residuals."""
        num_bits = self.cache_v_num_bits or self.num_bits
        self._validate_num_bits(num_bits)
        return num_bits

    @property
    def torch_scale_dtype(self) -> torch.dtype:
        """Torch dtype used for residual scale storage."""
        if self.scale_dtype == "bfloat16":
            return torch.bfloat16
        if self.scale_dtype == "float16":
            return torch.float16
        if self.scale_dtype == "float32":
            return torch.float32
        if self.scale_dtype == "float8_e4m3fn":
            if not hasattr(torch, "float8_e4m3fn"):
                raise ValueError("Current PyTorch build does not support float8_e4m3fn")
            return torch.float8_e4m3fn
        raise ValueError(f"Unsupported scale_dtype {self.scale_dtype!r}")

    def validate_kernel_impl(self) -> None:
        if self.kernel_impl not in ("native", "official_triton"):
            raise ValueError(f"Unsupported QVG kernel_impl {self.kernel_impl!r}")


class QVGBackend(KVStorageBackend):
    """QVG-style PRQ storage backend for dense K/V spans."""

    name = "qvg"

    def __init__(self, quant_config: QVGQuantConfig | None = None) -> None:
        self.quant_config = quant_config
        self.last_quantize_ms = 0.0
        self.last_dequantize_ms = 0.0
        self._compress_index = 0

    def _resolve_config(self, config: KVCompressionConfig) -> QVGQuantConfig:
        return self.quant_config or QVGQuantConfig.from_config(config)

    def compress_span(
        self,
        k: Tensor,
        v: Tensor,
        *,
        span: KVSpan,
        phase: RuntimePhase,
        config: KVCompressionConfig,
    ) -> KVStoragePayload:
        """Compress one K/V span in `[B, H, S, D]` layout."""
        if phase != RuntimePhase.FINALIZE_CLEAN_KV:
            raise ValueError("QVG compression must run after clean KV finalization")
        qcfg = self._resolve_config(config)
        qcfg.validate_kernel_impl()
        seed_base = _seed_for_compress_call(
            qcfg.kmeans_seed,
            compress_index=self._compress_index,
            num_stages=qcfg.num_prq_stages,
        )
        self._compress_index += 1
        start = time.perf_counter()
        quantize_tensor = (
            _official_prq_quantize_tensor
            if qcfg.kernel_impl == "official_triton"
            else _prq_quantize_tensor
        )
        official_kwargs = (
            {"preserve_rng": qcfg.preserve_rng}
            if qcfg.kernel_impl == "official_triton"
            else {}
        )
        k_state = quantize_tensor(
            k,
            num_stages=qcfg.num_prq_stages,
            num_clusters=qcfg.cache_num_k_centroids,
            max_iters=qcfg.kmeans_max_iters,
            block_size=qcfg.quant_block_size,
            num_bits=qcfg.k_num_bits,
            scale_dtype=qcfg.torch_scale_dtype,
            kmeans_seed=seed_base,
            **({"kmeans_init": qcfg.kmeans_init} if qcfg.kernel_impl == "native" else {}),
            **official_kwargs,
        )
        v_state = quantize_tensor(
            v,
            num_stages=qcfg.num_prq_stages,
            num_clusters=qcfg.cache_num_v_centroids,
            max_iters=qcfg.kmeans_max_iters,
            block_size=qcfg.quant_block_size,
            num_bits=qcfg.v_num_bits,
            scale_dtype=qcfg.torch_scale_dtype,
            kmeans_seed=(
                None if seed_base is None else seed_base + qcfg.num_prq_stages
            ),
            **({"kmeans_init": qcfg.kmeans_init} if qcfg.kernel_impl == "native" else {}),
            **official_kwargs,
        )
        self.last_quantize_ms = (time.perf_counter() - start) * 1000.0
        return KVStoragePayload(
            k=k_state,
            v=v_state,
            span=span,
            original_dtype=k.dtype,
            metadata={
                "backend": self.name,
                "quant_type": qcfg.quant_type,
                "num_bits": qcfg.num_bits,
                "k_num_bits": qcfg.k_num_bits,
                "v_num_bits": qcfg.v_num_bits,
                "kernel_impl": qcfg.kernel_impl,
                "quant_block_size": qcfg.quant_block_size,
                "quantize_ms": self.last_quantize_ms,
                "layout": "bhsd",
            },
        )

    def decompress_span(
        self,
        payload: KVStoragePayload,
        *,
        phase: RuntimePhase,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Decompress a QVG payload back to `[B, H, S, D]` layout."""
        del phase
        start = time.perf_counter()
        k_state = _move_tensor_tree(payload.k, device) if device is not None else payload.k
        v_state = _move_tensor_tree(payload.v, device) if device is not None else payload.v
        if payload.metadata.get("kernel_impl") == "official_triton":
            k = _official_prq_dequantize_tensor(
                k_state,
                output_dtype=payload.original_dtype,
            )
            v = _official_prq_dequantize_tensor(
                v_state,
                output_dtype=payload.original_dtype,
            )
        else:
            k = _prq_dequantize_tensor(k_state, output_dtype=payload.original_dtype)
            v = _prq_dequantize_tensor(v_state, output_dtype=payload.original_dtype)
        self.last_dequantize_ms = (time.perf_counter() - start) * 1000.0
        payload.metadata["dequantize_ms"] = self.last_dequantize_ms
        return k, v

    def estimate_bytes(self, payload: KVStoragePayload) -> int:
        """Estimate compressed payload storage bytes."""
        return estimate_tensor_tree_bytes(payload.k) + estimate_tensor_tree_bytes(
            payload.v
        )


def _seed_for_compress_call(
    kmeans_seed: int | None,
    *,
    compress_index: int,
    num_stages: int,
) -> int | None:
    """Derive a deterministic per-call seed while preserving official randomness.

    Official QVG uses the global CUDA RNG, so every layer sees different
    centroid samples. When a local seed is requested, we emulate that property
    without perturbing the model's global RNG state.
    """
    if kmeans_seed is None:
        return None
    return kmeans_seed + compress_index * num_stages * 2


def _prq_quantize_tensor(
    tensor: Tensor,
    *,
    num_stages: int,
    num_clusters: int,
    max_iters: int,
    block_size: int,
    num_bits: int,
    scale_dtype: torch.dtype,
    kmeans_init: str,
    kmeans_seed: int | None,
) -> dict[str, Any]:
    """Quantize one tensor in `[B, H, S, D]` layout."""
    assert tensor.ndim == 4, f"expected [B,H,S,D], got {tuple(tensor.shape)}"
    B, H, S, D = tensor.shape
    assert D % block_size == 0, (
        f"head_dim ({D}) must be divisible by quant_block_size ({block_size})"
    )
    assert num_clusters <= S, (
        f"num_clusters ({num_clusters}) must be <= sequence length ({S})"
    )

    residual = tensor if tensor.is_cuda else tensor.float()
    centroids_list: list[Tensor] = []
    cluster_ids_list: list[Tensor] = []

    for _stage in range(num_stages):
        cluster_ids, centroids = _batched_kmeans(
            residual,
            num_clusters=num_clusters,
            max_iters=max_iters,
            init=kmeans_init,
            seed=None if kmeans_seed is None else kmeans_seed + _stage,
        )
        centroids_list.append(centroids.to(dtype=tensor.dtype))
        cluster_ids_list.append(cluster_ids.to(torch.uint8))
        gathered = _gather_centroids(cluster_ids, centroids)
        residual = residual - gathered

    residual_quant, scales = _quantize_residual(
        residual,
        block_size=block_size,
        num_bits=num_bits,
        scale_dtype=scale_dtype,
    )
    return {
        "shape": (B, H, S, D),
        "centroids_list": centroids_list,
        "cluster_ids_list": cluster_ids_list,
        "residual_quant": residual_quant,
        "scales": scales,
        "block_size": block_size,
        "num_bits": num_bits,
        "kernel_impl": "native",
    }


def _official_prq_quantize_tensor(
    tensor: Tensor,
    *,
    num_stages: int,
    num_clusters: int,
    max_iters: int,
    block_size: int,
    num_bits: int,
    scale_dtype: torch.dtype,
    kmeans_seed: int | None,
    preserve_rng: bool,
) -> dict[str, Any]:
    from flashdreams.core.attention.kv_compress.qvg_official import (
        official_prq_quantize_tensor,
    )

    return official_prq_quantize_tensor(
        tensor,
        num_stages=num_stages,
        num_clusters=num_clusters,
        max_iters=max_iters,
        block_size=block_size,
        num_bits=num_bits,
        scale_dtype=scale_dtype,
        kmeans_seed=kmeans_seed,
        preserve_rng=preserve_rng,
    )


def _prq_dequantize_tensor(
    state: dict[str, Any],
    *,
    output_dtype: torch.dtype,
) -> Tensor:
    """Reconstruct one tensor from QVG PRQ state."""
    B, H, S, D = state["shape"]
    residual = _dequantize_residual(
        state["residual_quant"],
        state["scales"],
        block_size=state["block_size"],
        num_bits=state["num_bits"],
        D=D,
    )
    out = residual
    for cluster_ids, centroids in zip(
        state["cluster_ids_list"], state["centroids_list"], strict=True
    ):
        out = out + _gather_centroids(cluster_ids.long(), centroids.float())
    return out.reshape(B, H, S, D).to(dtype=output_dtype)


def _official_prq_dequantize_tensor(
    state: dict[str, Any],
    *,
    output_dtype: torch.dtype,
) -> Tensor:
    from flashdreams.core.attention.kv_compress.qvg_official import (
        official_prq_dequantize_tensor,
    )

    return official_prq_dequantize_tensor(state, output_dtype=output_dtype)


def _batched_kmeans(
    x: Tensor,
    *,
    num_clusters: int,
    max_iters: int,
    init: str,
    seed: int | None,
) -> tuple[Tensor, Tensor]:
    """Run k-means independently for each `[B,H]` stream."""
    B, H, S, D = x.shape
    BH = B * H
    x_flat = x.reshape(BH, S, D).contiguous()
    if not x_flat.is_cuda and x_flat.dtype != torch.float32:
        x_flat = x_flat.float()
    if init == "linspace":
        init_idx = torch.linspace(0, S - 1, num_clusters, device=x.device).round().long()
        init_idx = init_idx.expand(BH, num_clusters)
    elif init == "random":
        generator = None
        if seed is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed)
        init_idx = torch.randint(
            0,
            S,
            (BH, num_clusters),
            device=x.device,
            generator=generator,
        )
    else:
        raise ValueError(f"Unsupported kmeans_init {init!r}")
    centroids = torch.gather(
        x_flat,
        dim=1,
        index=init_idx.unsqueeze(-1).expand(-1, -1, D),
    ).clone()
    previous_labels: Tensor | None = None

    for _ in range(max_iters):
        distances = _squared_euclidean_distances(x_flat, centroids)
        new_labels = distances.argmin(dim=2)

        new_centroids = torch.zeros(
            BH,
            num_clusters,
            D,
            device=x.device,
            dtype=torch.float32,
        )
        index = new_labels.unsqueeze(-1).expand(BH, S, D)
        new_centroids.scatter_add_(dim=1, index=index, src=x_flat.float())

        counts = torch.zeros(
            BH,
            num_clusters,
            1,
            device=x.device,
            dtype=torch.float32,
        )
        counts.scatter_add_(
            dim=1,
            index=new_labels.unsqueeze(-1),
            src=torch.ones(BH, S, 1, device=x.device, dtype=torch.float32),
        )
        non_empty = counts > 0
        new_centroids = torch.where(
            non_empty,
            new_centroids / counts.clamp_min(1),
            centroids.float(),
        ).to(dtype=x_flat.dtype)
        if previous_labels is not None and torch.equal(new_labels, previous_labels):
            break
        previous_labels = new_labels
        centroids = new_centroids

    cluster_ids = new_labels.reshape(B, H, S)
    centroids = centroids.reshape(B, H, num_clusters, D)
    return cluster_ids, centroids


def _squared_euclidean_distances(x: Tensor, centroids: Tensor) -> Tensor:
    """Compute QVG-style squared Euclidean assignment distances."""
    x_sq = (x * x).sum(dim=-1).float()
    centroid_sq = (centroids * centroids).sum(dim=-1).float()
    cross = torch.bmm(x, centroids.transpose(1, 2)).float()
    distances = x_sq.unsqueeze(-1) + centroid_sq.unsqueeze(1) - 2.0 * cross
    return distances.clamp_min(0.0)


def _gather_centroids(cluster_ids: Tensor, centroids: Tensor) -> Tensor:
    """Gather centroid vector for each token assignment."""
    B, H, S = cluster_ids.shape
    D = centroids.shape[-1]
    index = cluster_ids.long().unsqueeze(-1).expand(B, H, S, D)
    return torch.gather(centroids, dim=2, index=index)


def _quantize_residual(
    residual: Tensor,
    *,
    block_size: int,
    num_bits: int,
    scale_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Blockwise signed residual quantization with packed UINT8 payload."""
    B, H, S, D = residual.shape
    max_int = 2 ** (num_bits - 1) - 1
    chunks = residual.float().reshape(B, H, S, D // block_size, block_size)
    scales = chunks.abs().amax(dim=-1).clamp_min(1e-10) / max_int
    q = _round_half_away_from_zero(chunks / scales.unsqueeze(-1)).clamp(
        -max_int, max_int
    )
    q = q.to(torch.int16).reshape(B, H, S, D)
    packed = _pack_signed(q, num_bits=num_bits)
    return packed, scales.to(dtype=scale_dtype)


def _round_half_away_from_zero(x: Tensor) -> Tensor:
    """Match Triton libdevice.round tie behavior used by official QVG."""
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _dequantize_residual(
    residual_quant: Tensor,
    scales: Tensor,
    *,
    block_size: int,
    num_bits: int,
    D: int,
) -> Tensor:
    q = _unpack_signed(residual_quant, num_bits=num_bits, D=D).float()
    B, H, S, _ = q.shape
    q = q.reshape(B, H, S, D // block_size, block_size)
    residual = q * scales.float().unsqueeze(-1)
    return residual.reshape(B, H, S, D)


def _pack_signed(q: Tensor, *, num_bits: int) -> Tensor:
    """Pack signed INT2/INT4 values into uint8 along the last dimension."""
    max_int = 2 ** (num_bits - 1) - 1
    unsigned = (q + max_int).to(torch.uint8)
    if num_bits == 4:
        assert unsigned.shape[-1] % 2 == 0, "INT4 packing requires even D"
        values = unsigned.reshape(*unsigned.shape[:-1], unsigned.shape[-1] // 2, 2)
        return ((values[..., 0] << 4) | values[..., 1]).contiguous()
    if num_bits == 2:
        assert unsigned.shape[-1] % 4 == 0, "INT2 packing requires D % 4 == 0"
        values = unsigned.reshape(*unsigned.shape[:-1], unsigned.shape[-1] // 4, 4)
        return (
            (values[..., 0] << 6)
            | (values[..., 1] << 4)
            | (values[..., 2] << 2)
            | values[..., 3]
        ).contiguous()
    raise ValueError(f"Unsupported num_bits {num_bits}")


def _unpack_signed(packed: Tensor, *, num_bits: int, D: int) -> Tensor:
    """Unpack uint8 INT2/INT4 values to signed int16."""
    max_int = 2 ** (num_bits - 1) - 1
    if num_bits == 4:
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        unsigned = torch.stack((high, low), dim=-1).reshape(*packed.shape[:-1], D)
    elif num_bits == 2:
        v0 = (packed >> 6) & 0x3
        v1 = (packed >> 4) & 0x3
        v2 = (packed >> 2) & 0x3
        v3 = packed & 0x3
        unsigned = torch.stack((v0, v1, v2, v3), dim=-1).reshape(
            *packed.shape[:-1], D
        )
    else:
        raise ValueError(f"Unsupported num_bits {num_bits}")
    return unsigned.to(torch.int16) - max_int


def _move_tensor_tree(value: Any, device: torch.device | str | None) -> Any:
    """Move tensors in a nested structure to ``device``."""
    if device is None:
        return value
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tensor_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tensor_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(v, device) for v in value)
    return value
