# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Quant-VideoGen:
# https://github.com/svg-project/Quant-VideoGen
# Source file: quant_videogen/kmeans/centroid_update.py
# Source commit: 11ff61a4f744d125a1d27fa13e2b9ecb89e94e07

import triton
import triton.language as tl
import torch
import torch.nn.functional as F


@triton.jit
def _centroid_update_kernel(
    x_ptr,  # *f16  [B, N, D]
    cluster_ptr,  # *i32  [B, N]
    sum_ptr,  # *f32  [B, K, D]
    count_ptr,  # *i32  [B, K]
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,  # number of dims processed per program
):
    """Each program processes 1 point (token) across BLOCK_D dimensions with atomics."""
    pid = tl.program_id(axis=0)
    token_idx = pid  # range: [0, B * N)

    # Derive (b, n) indices
    b = token_idx // N
    n = token_idx % N

    # Pointers to the token features and its cluster id
    x_offset = (b * N + n) * D
    x_ptr = x_ptr + x_offset

    cluster_idx = tl.load(cluster_ptr + b * N + n)  # int32

    # Guard for invalid cluster ids (should not happen)
    cluster_idx = tl.where(cluster_idx < K, cluster_idx, 0)

    # Base pointer for this centroid in the output sum tensor
    centroid_base = (b * K + cluster_idx) * D

    # Process feature vector in chunks of BLOCK_D
    offs = tl.arange(0, BLOCK_D)
    for d_start in range(0, D, BLOCK_D):
        mask = offs + d_start < D
        feats = tl.load(x_ptr + d_start + offs, mask=mask, other=0.0)
        feats = feats.to(tl.float32)

        dest_ptr = sum_ptr + centroid_base + d_start + offs
        tl.atomic_add(dest_ptr, feats, mask=mask)

    # Update counts (only once per point)
    tl.atomic_add(count_ptr + b * K + cluster_idx, 1)


def triton_centroid_update_cosine(
    x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor
):
    """Compute centroids using custom Triton kernel.

    Args:
        x_norm (Tensor): (B, N, D) normalized input vectors (float16/float32)
        cluster_ids (LongTensor): (B, N) cluster assignment per point
        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x_norm)

    Returns:
        Tensor: (B, K, D) updated and L2-normalized centroids (dtype == x_norm.dtype)
    """
    assert (
        x_norm.is_cuda and cluster_ids.is_cuda
    ), "Input tensors must be on CUDA device"
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # Allocate accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x_norm.device, dtype=torch.int32)

    # Launch Triton kernel – one program per token
    total_tokens = B * N
    BLOCK_D = 128  # tuneable

    grid = (total_tokens,)
    _centroid_update_kernel[grid](
        x_norm,
        cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        B,
        N,
        D,
        K,
        BLOCK_D=BLOCK_D,
    )

    # Compute means; keep old centroid if empty cluster
    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f

    # For clusters with zero count, revert to old centroids
    zero_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

    centroids = centroids.to(x_norm.dtype)
    centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids


def torch_loop_centroid_update_cosine(
    x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor
):
    """Reference Python implementation (double for-loop)"""
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    new_centroids = torch.zeros_like(old_centroids)
    for b in range(B):
        for k in range(K):
            mask = cluster_ids[b] == k
            if mask.any():
                new_centroids[b, k] = F.normalize(
                    x_norm[b][mask].mean(dim=0, dtype=x_norm.dtype), p=2, dim=0
                )
            else:
                new_centroids[b, k] = old_centroids[b, k]
    return new_centroids


def triton_centroid_update_euclid(
    x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor
):
    """Compute centroids for Euclidean KMeans using Triton.

    Args:
        x (Tensor): (B, N, D) input vectors (float16/float32)
        cluster_ids (LongTensor): (B, N) cluster assignment per point
        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x)

    Returns:
        Tensor: (B, K, D) updated centroids (dtype == x.dtype)
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # Allocate accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    total_tokens = B * N
    BLOCK_D = 128  # tuneable
    grid = (total_tokens,)

    _centroid_update_kernel[grid](
        x,
        cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        B,
        N,
        D,
        K,
        BLOCK_D=BLOCK_D,
    )

    # Compute means; keep old centroid if empty cluster
    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f

    # For clusters with zero count, revert to old centroids
    zero_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

    return centroids.to(x.dtype)


# ------------------------------ NEW: chunk-wise centroid update (sorted ids) ------------------------------


@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,  # *f16 / *f32 [B, N, D] – ORIGINAL ORDER
    sorted_idx_ptr,  # *i32        [B, N]    – indices after sort
    sorted_cluster_ptr,  # *i32        [B, N]    – cluster ids in sorted order
    sum_ptr,  # *f32        [B, K, D]
    count_ptr,  # *i32        [B, K]
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,  # how many tokens (points) each program processes
):
    """Each program processes **BLOCK_N consecutive, already-sorted tokens**.

    Because the tokens are sorted by cluster id, identical ids appear in
    contiguous runs.  We therefore accumulate a local sum/count for the
    current run and perform **a single atomic update per run**, instead of
    per-token.
    """
    # program indices – 2-D launch grid: (chunk_id, batch_id)
    pid_chunk = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    b = pid_b
    chunk_start = (
        pid_chunk * BLOCK_N
    )  # position of the first token handled by this program

    # Nothing to do – out of range
    if chunk_start >= N:
        return

    # base pointers for this batch
    idx_batch_base = sorted_idx_ptr + b * N
    cid_batch_base = sorted_cluster_ptr + b * N
    x_batch_base = x_ptr + b * N * D  # for pointer arithmetic

    # helper aranges
    offs_token = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, D)

    # first token index & validity mask
    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load first cluster id to initialise the running accumulator
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx, mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(
        idx_batch_base + token_idx, mask=valid_tok, other=-1
    )  # [BLOCK_N]

    load_mask = all_tokens_idxs[:, None] * D + offs_dim[None, :]

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            cluster_feats = tl.load(
                x_batch_base + load_mask, mask=cluster_mask[:, None], other=0.0
            )  # [BLOCK_N, D]
            cluster_feats = cluster_feats.to(tl.float32)
            sum_feats = tl.sum(cluster_feats, axis=0)
            dest_ptr = sum_ptr + (b * K + cid) * D + offs_dim
            tl.atomic_add(dest_ptr, sum_feats)
            tl.atomic_add(count_ptr + b * K + cid, cluster_size)


# ---------------------------------------------------------------------------------------------


def triton_centroid_update_sorted_euclid(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    num_centroids: int,
    *,
    BLOCK_N: int = 256,
):
    """Fast centroid update for *Euclidean* KMeans assuming cluster IDs are pre-sorted.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input feature vectors (no normalization assumed).
    cluster_ids : LongTensor [B, N]
        Cluster assignment for each point.
    num_centroids : int
        Number of centroids.
    BLOCK_N : int, optional
        Tokens per Triton program (affects occupancy/perf).
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    K = num_centroids

    # Batch-wise sort of cluster assignments
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    grid = (triton.cdiv(N, BLOCK_N), B)
    _centroid_update_chunk_kernel[grid](
        x,  # original features
        sorted_idx_int,  # gather indices
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        B,
        N,
        D,
        K,
        BLOCK_N=BLOCK_N,
    )

    # Convert sums to means; replace empty clusters with old centroids
    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    return centroids.to(x.dtype), centroid_cnts
