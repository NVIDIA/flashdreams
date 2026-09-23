// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "attention.cuh"

namespace omnidreams_singleview {

bool sage2_is_built() {
  return false;
}

bool sage2_is_runtime_supported(int /*device*/) {
  return false;
}

cudaError_t run_sage2_fmha_packed_qkv(
    const cutlass::bfloat16_t* /*Q*/,
    const cutlass::bfloat16_t* /*K*/,
    const cutlass::bfloat16_t* /*V*/,
    cutlass::bfloat16_t* /*O*/,
    int /*B*/, int /*Mq*/, int /*Mk*/,
    int /*H*/, int /*D*/,
    bool /*causal*/,
    float /*scale*/,
    cudaStream_t /*stream*/) {
  return cudaErrorNotSupported;
}

torch::Tensor sage2_test_attention(
    torch::Tensor /*q_bmhd*/,
    torch::Tensor /*k_bmhd*/,
    torch::Tensor /*v_bmhd*/,
    bool /*causal*/) {
  TORCH_CHECK(false,
              "SageAttention-2 was not built into OmniDreams single-view native extension");
}

}  // namespace omnidreams_singleview
