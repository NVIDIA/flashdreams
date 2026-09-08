// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <limits>

namespace omnidreams_singleview {

cudaError_t launch_sparge_attention_sm89_hd128(
    int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, half* O,
    int32_t* PV_Count, int32_t* __restrict__ Lut, int32_t* __restrict__ Valid_Block_Num,
    float* __restrict__ PV_Threshold,
    float* Q_scale, float* K_scale, float* V_scale,
    const uint32_t batch_size, const uint32_t qo_len, const uint32_t kv_len,
    const uint32_t num_qo_heads, const uint32_t num_kv_heads,
    const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
    const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
    const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
    const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
    float sm_scale, cudaStream_t stream);

}  // namespace omnidreams_singleview

namespace sparge_attention {

inline cudaError_t run_sparge_attention_sm89_hd128(
    int8_t* Q, int8_t* K, __nv_fp8_e4m3* V, half* O,
    int32_t* lut, int32_t* valid_block_num, float* pv_threshold,
    float* q_scale, float* k_scale, float* v_scale,
    int batch_size, int qo_len, int kv_len, int num_qo_heads, int num_kv_heads,
    int padded_kv_len, float sm_scale, cudaStream_t stream) {
  constexpr int D = 128;

  if (!Q || !K || !V || !O || !lut || !valid_block_num || !pv_threshold ||
      !q_scale || !k_scale || !v_scale) {
    return cudaErrorInvalidValue;
  }
  if (batch_size <= 0 || qo_len <= 0 || kv_len <= 0 ||
      num_qo_heads <= 0 || num_kv_heads <= 0 || padded_kv_len < kv_len) {
    return cudaErrorInvalidValue;
  }

  auto to_u32 = [](uint64_t value, uint32_t& out) -> bool {
    if (value > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
  };

  uint32_t stride_bz_q = 0;
  uint32_t stride_seq_q = 0;
  uint32_t stride_h_q = 0;
  uint32_t stride_bz_k = 0;
  uint32_t stride_seq_k = 0;
  uint32_t stride_h_k = 0;
  uint32_t stride_bz_v = 0;
  uint32_t stride_h_v = 0;
  uint32_t stride_d_v = 0;
  uint32_t stride_bz_o = 0;
  uint32_t stride_seq_o = 0;
  uint32_t stride_h_o = 0;

  if (!to_u32(uint64_t(qo_len) * num_qo_heads * D, stride_bz_q) ||
      !to_u32(uint64_t(num_qo_heads) * D, stride_seq_q) ||
      !to_u32(D, stride_h_q) ||
      !to_u32(uint64_t(kv_len) * num_kv_heads * D, stride_bz_k) ||
      !to_u32(uint64_t(num_kv_heads) * D, stride_seq_k) ||
      !to_u32(D, stride_h_k) ||
      !to_u32(uint64_t(num_kv_heads) * D * padded_kv_len, stride_bz_v) ||
      !to_u32(uint64_t(D) * padded_kv_len, stride_h_v) ||
      !to_u32(uint64_t(padded_kv_len), stride_d_v) ||
      !to_u32(uint64_t(qo_len) * num_qo_heads * D, stride_bz_o) ||
      !to_u32(uint64_t(num_qo_heads) * D, stride_seq_o) ||
      !to_u32(D, stride_h_o)) {
    return cudaErrorInvalidValue;
  }

  return omnidreams_singleview::launch_sparge_attention_sm89_hd128(
      Q, K, V, O,
      nullptr,  // PV_Count is unused by the instantiated return_pv_count=false kernel.
      lut, valid_block_num, pv_threshold,
      q_scale, k_scale, v_scale,
      batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      sm_scale, stream);
}

}  // namespace sparge_attention
