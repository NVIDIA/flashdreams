// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <atomic>

#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"

namespace omnidreams_singleview {

cudaError_t launch_sparge_attention_sm89_hd128(
    int8_t* Q,
    int8_t* K,
    __nv_fp8_e4m3* V,
    half* O,
    int32_t* PV_Count,
    int32_t* __restrict__ Lut,
    int32_t* __restrict__ Valid_Block_Num,
    float* __restrict__ PV_Threshold,
    float* Q_scale,
    float* K_scale,
    float* V_scale,
    const uint32_t batch_size,
    const uint32_t qo_len,
    const uint32_t kv_len,
    const uint32_t num_qo_heads,
    const uint32_t num_kv_heads,
    const uint32_t stride_bz_q,
    const uint32_t stride_seq_q,
    const uint32_t stride_h_q,
    const uint32_t stride_bz_k,
    const uint32_t stride_seq_k,
    const uint32_t stride_h_k,
    const uint32_t stride_bz_v,
    const uint32_t stride_h_v,
    const uint32_t stride_d_v,
    const uint32_t stride_bz_o,
    const uint32_t stride_seq_o,
    const uint32_t stride_h_o,
    float sm_scale,
    cudaStream_t stream) {
  if (batch_size == 0 || qo_len == 0 || kv_len == 0 ||
      num_qo_heads == 0 || num_kv_heads == 0 ||
      (num_qo_heads % num_kv_heads) != 0) {
    return cudaErrorInvalidValue;
  }

  constexpr uint32_t CTA_Q = 128;
  constexpr uint32_t CTA_K = 64;
  constexpr uint32_t WARP_Q = 32;
  constexpr uint32_t WARP_K = 64;
  constexpr uint32_t kHeadDim = 128;
  constexpr MaskMode mask_mode = MaskMode::kNone;

  size_t smem_max = std::max(
      CTA_Q * kHeadDim * sizeof(int8_t) +
          CTA_K * kHeadDim * sizeof(int8_t) +
          CTA_K * kHeadDim * sizeof(int8_t),
      CTA_Q * kHeadDim * sizeof(half));
  auto kernel_func = qk_int_sv_f8_block_sparse_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, kHeadDim, DataType::kInt8,
      QuantGranularity::kPerBlock, QuantGranularity::kPerBlock, float,
      true, false, PVThresholdMode::kPerBlock, half, ComputeUnit::kCudaCore,
      mask_mode, true, false>;

  static std::atomic<bool> smem_attr_set{false};
  if (!smem_attr_set.load(std::memory_order_acquire)) {
    cudaError_t err = cudaFuncSetAttribute(
        kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);
    if (err != cudaSuccess) {
      return err;
    }
    smem_attr_set.store(true, std::memory_order_release);
  }

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  kernel_func<<<grid, block, smem_max, stream>>>(
      Q,
      K,
      reinterpret_cast<int8_t*>(V),
      O,
      PV_Count,
      Lut,
      Valid_Block_Num,
      PV_Threshold,
      Q_scale,
      K_scale,
      V_scale,
      qo_len,
      kv_len,
      num_qo_heads / num_kv_heads,
      stride_bz_q,
      stride_seq_q,
      stride_h_q,
      stride_bz_k,
      stride_seq_k,
      stride_h_k,
      stride_bz_v,
      stride_h_v,
      stride_d_v,
      stride_bz_o,
      stride_seq_o,
      stride_h_o,
      sm_scale);

  return cudaGetLastError();
}

}  // namespace omnidreams_singleview
