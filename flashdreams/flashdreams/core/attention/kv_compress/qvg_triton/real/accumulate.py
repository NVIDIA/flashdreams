# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Quant-VideoGen:
# https://github.com/svg-project/Quant-VideoGen
# Source file: quant_videogen/real/accumulate.py
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
                    triton.Config({"BLOCK_S": BLOCK_S}, num_stages=num_stages, num_warps=num_warps)
                )
    return configs


@triton.autotune(
    configs=get_configs(),
    key=["S", "D"],
)
@triton.jit
def _nstage_accum_kernel(
    residual_quant_ptr,
    scales_ptr,
    centroids_ptr,
    cluster_ids_ptr,
    output_ptr,
    # Dimensions
    S: tl.constexpr,
    D: tl.constexpr,
    D_PACKED: tl.constexpr,  # D // pack_factor (or D if not packed)
    SCALE_D: tl.constexpr,  # D // block_size
    N_STAGES: tl.constexpr,
    K: tl.constexpr,  # n_clusters per stage
    # Quantization params
    n_bits: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    PACK_INPUT_INT8: tl.constexpr,  # Whether input is packed into int8
    # Autotune
    BLOCK_S: tl.constexpr,
):
    """
    Accumulate multi-stage KMeans centroids and dequantized residual.
    
    Layout: (B*H, S, D) - no reshape needed, direct from input format.
    Grid: (B*H, cdiv(S, BLOCK_S))
    
    Each program handles BLOCK_S tokens for one (b, h) pair.
    
    Steps:
    1. Unpack (if packed) and dequantize residual
    2. For each stage, load cluster_id and gather corresponding centroid
    3. Accumulate all centroids + dequantized residual
    """
    pid_bh = tl.program_id(0)  # (b, h) index in [0, B*H)
    pid_s = tl.program_id(1)   # S block index
    
    # Offsets
    offset_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offset_d = tl.arange(0, D)
    offset_d_packed = tl.arange(0, D_PACKED)
    offset_scale_d = tl.arange(0, SCALE_D)
    
    mask_s = offset_s < S
    
    # ==================== Step 1: Dequantize residual ====================
    max_int_value = 2**(n_bits - 1) - 1
    
    if PACK_INPUT_INT8:
        # Load packed quantized values: [BH, S, D_packed]
        residual_ptr = residual_quant_ptr + pid_bh * S * D_PACKED + offset_s[:, None] * D_PACKED + offset_d_packed[None, :]
        residual_packed = tl.load(residual_ptr, mask=mask_s[:, None], other=0)
        residual_packed = residual_packed.to(tl.int32)
        
        # tl.device_print("pid_bh: ", pid_bh)
        # tl.device_print("pid_s: ", pid_s)
        # tl.device_print("residual_packed: ", residual_packed)
        
        
        # Unpack based on n_bits
        if n_bits == 4:
            # Packing: y1 << 4 | y2, y1=even, y2=odd
            high = (residual_packed >> 4) & 0xF
            low = residual_packed & 0xF

            # Join to (BLOCK_S, D_PACKED, 2) then reshape to (BLOCK_S, D)
            residual_unpacked = tl.reshape(tl.join(high, low), (BLOCK_S, D))
            residual_unpacked = residual_unpacked - max_int_value
        elif n_bits == 2:
            v1 = (residual_packed >> 6) & 0x3
            v2 = (residual_packed >> 4) & 0x3
            v3 = (residual_packed >> 2) & 0x3
            v4 = residual_packed & 0x3
            v13 = tl.join(v1, v3)
            v24 = tl.join(v2, v4)
            v13 = tl.reshape(v13, (BLOCK_S, D // 2))
            v24 = tl.reshape(v24, (BLOCK_S, D // 2))
            residual_unpacked = tl.reshape(tl.join(v13, v24), (BLOCK_S, D))
            residual_unpacked = residual_unpacked - max_int_value
        else:
            # n_bits == 8, packed as int8 (no bit packing, just dtype)
            residual_ptr_8 = residual_quant_ptr + pid_bh * S * D + offset_s[:, None] * D + offset_d[None, :]
            residual_unpacked = tl.load(residual_ptr_8, mask=mask_s[:, None], other=0)
    else:
        # Not packed: residual is stored as int8 with shape [BH, S, D]
        residual_ptr = residual_quant_ptr + pid_bh * S * D + offset_s[:, None] * D + offset_d[None, :]
        residual_unpacked = tl.load(residual_ptr, mask=mask_s[:, None], other=0)
        
    # tl.device_print("After unpacking: ", residual_unpacked)
    # tl.static_print("After unpacking: ", residual_unpacked)
    
    residual_unpacked = residual_unpacked.to(tl.float32)
    
    # Load scales: [BH, S, SCALE_D]
    scale_ptr = scales_ptr + pid_bh * S * SCALE_D + offset_s[:, None] * SCALE_D + offset_scale_d[None, :]
    scales = tl.load(scale_ptr, mask=mask_s[:, None], other=1.0)
    scales = scales.to(tl.float32)
    
    # Dequantize: multiply by scale (broadcast along Q_BLOCK_SIZE)
    # residual_unpacked: (BLOCK_S, D), scales: (BLOCK_S, SCALE_D)
    # Each scale covers Q_BLOCK_SIZE elements
    residual_reshaped = tl.reshape(residual_unpacked, (BLOCK_S, SCALE_D, Q_BLOCK_SIZE))
    scales_expanded = tl.reshape(scales, (BLOCK_S, SCALE_D, 1))
    residual_dequant = residual_reshaped * scales_expanded
    residual_dequant = tl.reshape(residual_dequant, (BLOCK_S, D))
    
    # Initialize accumulator
    accum = residual_dequant
    
    # ==================== Step 2 & 3: Accumulate centroids ====================
    # For each stage, load cluster_id and add corresponding centroid
    for stage in range(N_STAGES):
        # Load cluster_ids: [BH, N_STAGES, S]
        cluster_id_ptr = cluster_ids_ptr + pid_bh * N_STAGES * S + stage * S + offset_s
        cluster_id = tl.load(cluster_id_ptr, mask=mask_s, other=0).to(tl.int32)  # (BLOCK_S,)

        # Load centroids: [BH, N_STAGES * K, D]
        # Offset: pid_bh * (N_STAGES * K * D) + (stage * K + cluster_id) * D + d
        centroid_base = centroids_ptr + pid_bh * (N_STAGES * K * D) + stage * K * D
        centroid_offset = centroid_base + cluster_id[:, None] * D + offset_d[None, :]
        centroid = tl.load(centroid_offset, mask=mask_s[:, None], other=0.0)  # (BLOCK_S, D)
        
        accum = accum + centroid
    
    # ==================== Store output ====================
    # Output: [BH, S, D]
    output_offset = output_ptr + pid_bh * S * D + offset_s[:, None] * D + offset_d[None, :]
    tl.store(output_offset, accum, mask=mask_s[:, None])


def nstage_accum(
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
    Accumulate multi-stage KMeans centroids and quantized residual to reconstruct x.

    Reconstructs the original tensor by:
      1. Dequantizing the residual using scales
      2. For each stage, adding back the corresponding centroid based on cluster_ids
      3. Accumulating all contributions to recover the original distribution

    Args:
        centroids_list: List of n_stages tensors, each of shape (B, H, n_clusters, D),
                        the centroids from each stage.
        cluster_ids_list: List of n_stages tensors, each of shape (B, H, S),
                          the cluster assignments from each stage.
        residual_quant: Quantized residual tensor.
                        If PACK_INPUT_INT8=True: shape (B, H, S, D // pack_factor) as uint8,
                        where pack_factor = 8 // num_bits.
                        If PACK_INPUT_INT8=False: shape (B, H, S, D) as int8.
        scales: Scale factors from quantization, shape (B, H, S, D // block_size).
        block_size: Block size used in quantization.
        num_bits: Number of bits used in quantization (2, 4, or 8).
        PACK_INPUT_INT8: If True, residual is packed into uint8 (multiple values per byte).
                         If False, residual is stored as int8 (one value per byte).

    Returns:
        x_reconstructed: Tensor of shape (B, H, S, D), the reconstructed tensor.
                         x_reconstructed ≈ sum(centroids[stage][cluster_ids[stage]]) + dequant(residual)
    """
    assert num_bits in (2, 4, 8), "num_bits must be 2, 4, or 8"
    assert len(centroids_list) == len(cluster_ids_list), "centroids_list and cluster_ids_list must have same length"
    if PACK_INPUT_INT8:
        assert num_bits in (2, 4), "PACK_INPUT_INT8=True requires num_bits in (2, 4)"

    N_STAGES = len(centroids_list)
    B, H, K, D = centroids_list[0].shape
    S = cluster_ids_list[0].shape[2]
    BH = B * H
    
    # Determine pack factor and D_PACKED
    if PACK_INPUT_INT8 and num_bits in (2, 4):
        pack_factor = 8 // num_bits
        D_PACKED = D // pack_factor
    else:
        D_PACKED = D  # Not packed, one value per byte
    
    SCALE_D = D // block_size
    
    # ==================== Flatten to (B*H, ...) layout - no permute needed ====================
    # Residual: (B, H, S, D_packed) -> (B*H, S, D_packed)
    residual_flat = residual_quant.reshape(BH, S, D_PACKED).contiguous()
    
    # Scales: (B, H, S, D // block_size) -> (B*H, S, D // block_size)
    scales_flat = scales.reshape(BH, S, SCALE_D).contiguous()
    
    # Centroids: list of (B, H, K, D) -> concat to (B*H, N_STAGES * K, D)
    # Stack along a new dim then reshape
    centroids_stacked = torch.stack(centroids_list, dim=2)  # (B, H, N_STAGES, K, D)
    centroids_flat = centroids_stacked.reshape(BH, N_STAGES * K, D).contiguous()
    
    # Cluster IDs: list of (B, H, S) -> stack to (B, H, N_STAGES, S) -> (B*H, N_STAGES, S)
    cluster_ids_stacked = torch.stack(cluster_ids_list, dim=2)  # (B, H, N_STAGES, S)
    cluster_ids_flat = cluster_ids_stacked.reshape(BH, N_STAGES, S).contiguous()
    
    # Output: (B*H, S, D)
    output = torch.empty(BH, S, D, device=residual_quant.device, dtype=output_dtype)

    # ==================== Launch kernel ====================
    grid = lambda META: (BH, triton.cdiv(S, META["BLOCK_S"]))
    
    _nstage_accum_kernel[grid](
        residual_flat,
        scales_flat,
        centroids_flat,
        cluster_ids_flat,
        output,
        S,
        D,
        D_PACKED,
        SCALE_D,
        N_STAGES,
        K,
        num_bits,
        block_size,
        PACK_INPUT_INT8,
    )
    
    # Reshape output back to (B, H, S, D)
    output = output.reshape(B, H, S, D)
    
    return output
