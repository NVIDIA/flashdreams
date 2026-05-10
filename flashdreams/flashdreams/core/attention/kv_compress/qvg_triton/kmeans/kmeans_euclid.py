# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Quant-VideoGen:
# https://github.com/svg-project/Quant-VideoGen
# Source file: quant_videogen/kmeans/kmeans_euclid.py
# Source commit: 11ff61a4f744d125a1d27fa13e2b9ecb89e94e07

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ..timer import time_logging_decorator

from .euclid_assign import euclid_assign_triton
from .centroid_update import triton_centroid_update_sorted_euclid


# 1. Euclidean
@time_logging_decorator("Level 5 - batch kmeans euclid iter")
def _euclid_iter(x, x_sq, centroids):
    cluster_ids = euclid_assign_triton(x, centroids, x_sq)

    num_centroids = centroids.shape[1]
    centroids_new, cluster_sizes = triton_centroid_update_sorted_euclid(
        x, cluster_ids, num_centroids
    )

    empty_mask = (cluster_sizes == 0).unsqueeze(-1)
    centroids_new = torch.where(empty_mask, centroids, centroids_new)

    return centroids_new, cluster_ids, cluster_sizes


@time_logging_decorator("Level 4 - batch kmeans euclid")
def batch_kmeans_Euclid(
    x, n_clusters, max_iters=100, tol=1e-4, init_centroids=None, verbose=False
):
    """
    Batched KMeans clustering in PyTorch using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
        cluster_sizes: (B, n_clusters) LongTensor, number of points per cluster.
        n_iters: actual number of iterations executed (int)
    """
    B, N, D = x.shape

    # Pre-compute squared L2 norm of all points (constant during iterations)
    x_sq = (x**2).sum(dim=-1)  # (B, N)

    if init_centroids is None:
        # Randomly select initial centers from x
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x, dim=1, index=indices[..., None].expand(-1, -1, D)
        )  # (B, n_clusters, D)
    else:
        # centroids = init_centroids.clone()
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    prev_cluster_ids = None

    for it in range(max_iters):
        # ---- compiled single iteration ----

        centroids_new, cluster_ids, cluster_sizes = _euclid_iter(x, x_sq, centroids)

        if prev_cluster_ids is not None:
            # Check for convergence, given the tolenrance
            token_num = cluster_ids.numel()
            change_num = (cluster_ids != prev_cluster_ids).sum()
            if verbose:
                # if True:
                print(
                    f"Iter {it}, token num: {token_num} | change num: {change_num} | change rate: {change_num / token_num}."
                )
            if change_num / token_num < tol:
                break

        prev_cluster_ids = cluster_ids
        centroids = centroids_new

    return cluster_ids, centroids, cluster_sizes, it + 1
    # return cluster_ids.clone(), centroids.clone(), cluster_sizes.clone(), it + 1


# batch_kmeans_Euclid = torch.compile(batch_kmeans_Euclid, dynamic=True, mode="reduce-overhead")
