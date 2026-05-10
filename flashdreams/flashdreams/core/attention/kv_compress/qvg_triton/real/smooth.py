# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Quant-VideoGen:
# https://github.com/svg-project/Quant-VideoGen
# Source file: quant_videogen/real/smooth.py
# Source commit: 11ff61a4f744d125a1d27fa13e2b9ecb89e94e07

import torch
import triton
import triton.language as tl

########################################################
# Triton kernels
########################################################


def get_configs():
    configs = []
    for BLOCK_S in [32, 64, 128]:
        for num_warps in [4, 8]:
            for num_stages in [3, 4, 5]:
                configs.append(
                    triton.Config(
                        {"BLOCK_S": BLOCK_S}, num_stages=num_stages, num_warps=num_warps
                    )
                )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=["N", "D"],
)
@triton.jit
def _minus_centroid_kernel(
    X_ptr,
    cluster_ids_ptr,
    centroids_ptr,
    Y_ptr,
    S: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Subtract the corresponding centroid from each point based on cluster assignment.
    """
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    offset_d = tl.arange(0, D)
    offset_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    # Load cluster id
    cluster_id_ptr = cluster_ids_ptr + pid_bh * S + offset_s
    cluster_id = tl.load(cluster_id_ptr, mask=offset_s < S, other=0).to(tl.int32)
    
    # Load centroids
    centroids_ptr = centroids_ptr + pid_bh * K * D + cluster_id[:, None] * D + offset_d[None, :]
    centroids = tl.load(centroids_ptr)
    
    # Load X
    x_ptr = X_ptr + pid_bh * S * D + offset_s[:, None] * D + offset_d[None, :]
    x = tl.load(x_ptr, mask=offset_s[:, None] < S, other=0.0)
    
    # Subtract centroids
    y = x - centroids

    # Store result
    y_ptr = Y_ptr + pid_bh * S * D + offset_s[:, None] * D + offset_d[None, :]

    tl.store(y_ptr, y, mask=offset_s[:, None] < S)


def minus_centroid(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """
    Subtract the corresponding centroid from each point based on cluster assignment.

    This function takes the output of batch_kmeans_Euclid and computes residuals
    by subtracting each point's assigned centroid from the point itself.

    Args:
        x: Tensor of shape (B, H, S, D), the original input points.
        cluster_ids: Tensor of shape (B, H, S), cluster assignment for each point
                     (output from batch_kmeans_Euclid).
        centroids: Tensor of shape (B, H, n_clusters, D), the cluster centers
                   (output from batch_kmeans_Euclid).

    Returns:
        residuals: Tensor of shape (B, H, S, D), where each point has its
                   corresponding centroid subtracted.
    """
    B, H, S, D = x.shape
    K = centroids.shape[2]
    assert cluster_ids.shape == (B, H, S), f"cluster_ids shape mismatch. cluster_ids.shape: {cluster_ids.shape}, expected: ({B}, {H}, {S})"
    assert centroids.shape == (B, H, K, D), f"centroids shape mismatch. centroids.shape: {centroids.shape}, expected: ({B}, {H}, {K}, {D})"

    BH = B * H

    # Flatten the tensor
    x_flat = x.reshape(BH, S, D).contiguous()
    cluster_ids_flat = cluster_ids.reshape(BH, S).contiguous()
    centroids_flat = centroids.reshape(BH, K, D).contiguous()
    residuals_flat = torch.empty_like(x_flat)

    grid = lambda META: (BH, triton.cdiv(S, META["BLOCK_S"]))
    _minus_centroid_kernel[grid](
        x_flat,
        cluster_ids_flat,
        centroids_flat,
        residuals_flat,
        S,
        K,
        D,
    )
    return residuals_flat.reshape(B, H, S, D)
