# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for vendored official Quant-VideoGen Triton PRQ kernels."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor


def _fork_devices(tensor: Tensor) -> list[int]:
    if not tensor.is_cuda:
        return []
    index = tensor.device.index
    if index is None:
        index = torch.cuda.current_device()
    return [index]


def _validate_scale_dtype_for_device(tensor: Tensor, scale_dtype: torch.dtype) -> None:
    if not tensor.is_cuda:
        return
    if scale_dtype != getattr(torch, "float8_e4m3fn", None):
        return
    major, _minor = torch.cuda.get_device_capability(tensor.device)
    if major < 9:
        raise ValueError(
            "Official QVG Triton FP8 scale path does not compile on this GPU. "
            "Use scale_dtype='bfloat16' for official_triton on SM80/Ampere."
        )


def official_prq_quantize_tensor(
    tensor: Tensor,
    *,
    num_stages: int,
    num_clusters: int,
    max_iters: int,
    block_size: int,
    num_bits: int,
    scale_dtype: torch.dtype,
    kmeans_seed: int | None,
    preserve_rng: bool = True,
) -> dict[str, Any]:
    """Quantize one `[B,H,S,D]` tensor with vendored official QVG kernels."""
    from flashdreams.core.attention.kv_compress.qvg_triton.real.prq import prq_quant

    _validate_scale_dtype_for_device(tensor, scale_dtype)
    B, H, S, D = tensor.shape
    devices = _fork_devices(tensor)
    rng_context = (
        torch.random.fork_rng(devices=devices) if preserve_rng else nullcontext()
    )
    with rng_context:
        if kmeans_seed is not None:
            torch.manual_seed(kmeans_seed)
        centroids_list, cluster_ids_list, residual_quant, scales = prq_quant(
            tensor.contiguous(),
            n_stages=num_stages,
            n_clusters=num_clusters,
            block_size=block_size,
            num_bits=num_bits,
            scale_precision=scale_dtype,
            max_iters=max_iters,
            PACK_OUTPUT_INT8=True,
            CLUSTER_ID_INT8=True,
        )
    return {
        "shape": (B, H, S, D),
        "centroids_list": centroids_list,
        "cluster_ids_list": cluster_ids_list,
        "residual_quant": residual_quant,
        "scales": scales,
        "block_size": block_size,
        "num_bits": num_bits,
        "kernel_impl": "official_triton",
    }


def official_prq_dequantize_tensor(
    state: dict[str, Any],
    *,
    output_dtype: torch.dtype,
) -> Tensor:
    """Dequantize official QVG PRQ state with vendored Triton accumulation."""
    from flashdreams.core.attention.kv_compress.qvg_triton.real.prq import (
        prq_dequant,
    )

    return prq_dequant(
        centroids_list=state["centroids_list"],
        cluster_ids_list=state["cluster_ids_list"],
        residual_quant=state["residual_quant"],
        scales=state["scales"],
        block_size=state["block_size"],
        num_bits=state["num_bits"],
        PACK_INPUT_INT8=True,
        CLUSTER_ID_INT8=True,
        output_dtype=output_dtype,
    )
