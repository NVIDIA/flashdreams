# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Quant-VideoGen:
# https://github.com/svg-project/Quant-VideoGen
# Source file: quant_videogen/real/prq.py
# Source commit: 11ff61a4f744d125a1d27fa13e2b9ecb89e94e07

import torch

from ..kmeans.kmeans_euclid import batch_kmeans_Euclid

from .smooth import minus_centroid
from .quant_pack import quant_pack
from .accumulate import nstage_accum


def prq_quant(
    x: torch.Tensor,
    n_stages: int,
    n_clusters: int,
    block_size: int,
    num_bits: int,
    scale_precision: torch.dtype,
    max_iters: int = 100,
    tol: float = 1e-4,
    PACK_OUTPUT_INT8: bool = False,
    CLUSTER_ID_INT8: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Multi-stage KMeans quantization pipeline.

    Performs n_stages iterations of:
      1. KMeans clustering to find centroids
      2. Subtract centroids from x (minus_centroid)
      3. Continue with residuals for next stage
    After all stages, quantize the final residual using quant_pack.

    Args:
        x: Tensor of shape (B, H, S, D), the input tensor to quantize.
        n_stages: Number of KMeans stages to perform.
        n_clusters: Number of clusters for each KMeans stage.
        block_size: Block size for final residual quantization.
        num_bits: Number of bits for quantization.
            - Must be 2, 3, 4, or 8.
        scale_precision: Precision for scale factors.
            - Must be torch.bfloat16 or torch.float8_e4m3fn.
        max_iters: Max iterations for each KMeans.
        tol: Tolerance for KMeans convergence.
        PACK_OUTPUT_INT8: If True, pack quantized output into uint8.
        CLUSTER_ID_INT8: If True, cluster ids are stored as uint8.
    Returns:
        centroids_list: List of n_stages tensors, each of shape (B, H, n_clusters, D),
                        the centroids from each stage.
        cluster_ids_list: List of n_stages tensors, each of shape (B, H, S),
                          the cluster assignments from each stage.
        residual_quant: Quantized final residual tensor.
        scales: Scale factors from quantization.
    """
    B, H, S, D = x.shape
    BH = B * H
    
    centroids_list = []
    cluster_ids_list = []
    
    residual = x
    
    # Multi-stage KMeans: iteratively cluster and subtract centroids
    for stage in range(n_stages):
        # Run KMeans on flattened (B*H, S, D) tensor
        cluster_ids, centroids, cluster_sizes, iters = batch_kmeans_Euclid(
            residual.reshape(BH, S, D),
            n_clusters=n_clusters,
            max_iters=max_iters,
            tol=tol,
        )
        
        # Reshape back to (B, H, ...)
        cluster_ids = cluster_ids.reshape(B, H, S)
        centroids = centroids.reshape(B, H, n_clusters, D)
        
        if CLUSTER_ID_INT8:
            assert cluster_ids.max() < 256, "Cluster ids must be less than 256 when CLUSTER_ID_INT8 is True"
            cluster_ids = cluster_ids.to(torch.uint8)
        
        centroids_list.append(centroids)
        cluster_ids_list.append(cluster_ids)
        
        # Subtract centroids to get residual for next stage
        residual = minus_centroid(residual, cluster_ids, centroids)

    # Quantize final residual
    residual_quant, scales = quant_pack(
        residual,
        block_size=block_size,
        num_bits=num_bits,
        scale_precision=scale_precision,
        pack_output_int8=PACK_OUTPUT_INT8,
    )
    
    return centroids_list, cluster_ids_list, residual_quant, scales


def prq_dequant(
    centroids_list: list[torch.Tensor],
    cluster_ids_list: list[torch.Tensor],
    residual_quant: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    num_bits: int = 4,
    PACK_INPUT_INT8: bool = False,
    CLUSTER_ID_INT8: bool = False,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize and reconstruct tensor from multi-stage KMeans quantization.

    Reconstructs the original tensor by:
      1. Dequantizing the residual using scales
      2. For each stage, adding back the corresponding centroid based on cluster_ids
      3. Accumulating all contributions to recover the original distribution

    Args:
        centroids_list: List of n_stages tensors, each of shape (B, H, n_clusters, D),
                        the centroids from each stage.
        cluster_ids_list: List of n_stages tensors, each of shape (B, H, S),
                          the cluster assignments from each stage.
        residual_quant: Quantized residual tensor from prq_quant.
        scales: Scale factors from quantization, shape (B, H, S, D // block_size).
        block_size: Block size used in quantization.
        num_bits: Number of bits used in quantization (2, 4, or 8).
        PACK_INPUT_INT8: If True, residual is packed into uint8.

    Returns:
        x_reconstructed: Tensor of shape (B, H, S, D), the reconstructed tensor.
    """
    return nstage_accum(
        centroids_list=centroids_list,
        cluster_ids_list=cluster_ids_list,
        residual_quant=residual_quant,
        scales=scales,
        block_size=block_size,
        num_bits=num_bits,
        PACK_INPUT_INT8=PACK_INPUT_INT8,
        CLUSTER_ID_INT8=CLUSTER_ID_INT8,
        output_dtype=output_dtype,
    )
