// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "attention.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdio>

#include "csrc/fused/fused.h"
#include "csrc/qattn/attn_cuda_sm90.h"

namespace omnidreams_singleview {
namespace {

struct CurrentStreamScope {
  explicit CurrentStreamScope(cudaStream_t stream, int device)
      : previous_(c10::cuda::getCurrentCUDAStream(device)),
        external_(c10::cuda::getStreamFromExternal(stream, device)) {
    c10::cuda::setCurrentCUDAStream(external_);
  }

  ~CurrentStreamScope() {
    c10::cuda::setCurrentCUDAStream(previous_);
  }

  c10::cuda::CUDAStream previous_;
  c10::cuda::CUDAStream external_;
};

at::Tensor packed_bmhk_view(const cutlass::bfloat16_t* ptr,
                            int B, int M, int H, int D, int device) {
  auto opts = at::TensorOptions()
                  .device(at::kCUDA, device)
                  .dtype(at::kBFloat16);
  return torch::from_blob(
      const_cast<cutlass::bfloat16_t*>(ptr),
      {B, M, H, D},
      opts);
}

at::Tensor packed_bmhk_output_view(cutlass::bfloat16_t* ptr,
                                   int B, int M, int H, int D, int device) {
  auto opts = at::TensorOptions()
                  .device(at::kCUDA, device)
                  .dtype(at::kBFloat16);
  return torch::from_blob(ptr, {B, M, H, D}, opts);
}

int round_up(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

cudaError_t sage2_check_last_error(const char* stage) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "OmniDreams Native Sage2 failed after %s: %s\n",
                 stage, cudaGetErrorString(err));
  }
  return err;
}

}  // namespace

bool sage2_is_built() {
  return true;
}

bool sage2_is_runtime_supported(int device) {
  if (device < 0) {
    if (cudaGetDevice(&device) != cudaSuccess) return false;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return false;
  return prop.major == 9 && prop.minor == 0;
}

cudaError_t run_sage2_fmha_packed_qkv(
    const cutlass::bfloat16_t* Q,
    const cutlass::bfloat16_t* K,
    const cutlass::bfloat16_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream) {
  if (!Q || !K || !V || !O || B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0) {
    return cudaErrorInvalidValue;
  }
  if (D != 64 && D != 128) return cudaErrorNotSupported;

  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) return cudaErrorInvalidDevice;
  if (!sage2_is_runtime_supported(device)) return cudaErrorNotSupported;

  CurrentStreamScope stream_scope(stream, device);
  const auto q = packed_bmhk_view(Q, B, Mq, H, D, device);
  const auto k = packed_bmhk_view(K, B, Mk, H, D, device);
  const auto v = packed_bmhk_view(V, B, Mk, H, D, device);
  auto out = packed_bmhk_output_view(O, B, Mq, H, D, device);

  // Match SageAttention-2's SM90 per-warp path in NHD layout. K smoothing is
  // mathematically output-preserving and improves INT8 quantization accuracy.
  auto q_int8 = at::empty(q.sizes(), q.options().dtype(at::kChar));
  auto k_int8 = at::empty(k.sizes(), k.options().dtype(at::kChar));
  auto q_scale = at::empty(
      {B, H, ((Mq + 63) / 64) * 4}, q.options().dtype(at::kFloat));
  auto k_scale = at::empty(
      {B, H, (Mk + 127) / 128}, k.options().dtype(at::kFloat));
  auto k_mean = k.mean(/*dim=*/1, /*keepdim=*/false);

  quant_per_warp_int8_cuda(
      q, q_int8, q_scale,
      /*block_size=*/64, /*warp_block_size=*/16, /*NHD=*/0);
  if (cudaError_t err = sage2_check_last_error("Q quantization");
      err != cudaSuccess) {
    return err;
  }
  quant_per_block_int8_fuse_sub_mean_cuda(
      k, k_mean, k_int8, k_scale, /*block_size=*/128, /*NHD=*/0);
  if (cudaError_t err = sage2_check_last_error("K quantization");
      err != cudaSuccess) {
    return err;
  }

  const int padded_mk = round_up(Mk, 128);
  at::Tensor v_padded = v;
  if (padded_mk != Mk) {
    v_padded = at::zeros({B, padded_mk, H, D}, v.options());
    v_padded.narrow(/*dim=*/1, /*start=*/0, /*length=*/Mk).copy_(v);
  }
  auto v_transposed = at::empty(
      {B, D, H, padded_mk}, v.options());
  auto v_fp8 = at::empty(
      v_transposed.sizes(), v.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto v_scale = at::empty({B, H, D}, v.options().dtype(at::kFloat));
  transpose_pad_permute_cuda(v_padded, v_transposed, /*NHD=*/0);
  if (cudaError_t err = sage2_check_last_error("V transpose/pad");
      err != cudaSuccess) {
    return err;
  }
  scale_fuse_quant_cuda(
      v_transposed, v_fp8, v_scale, padded_mk,
      /*E4M3 max=*/448.0f, /*NHD=*/0);
  if (cudaError_t err = sage2_check_last_error("V quantization");
      err != cudaSuccess) {
    return err;
  }

  const float sm_scale = scale == 0.0f ? 1.0f / std::sqrt(float(D)) : scale;
  (void)qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
      q_int8, k_int8, v_fp8, out, q_scale, k_scale, v_scale,
      /*NHD=*/0, causal ? 1 : 0,
      /*per-warp=*/2, sm_scale,
      /*return_lse=*/0);
  return sage2_check_last_error("attention");
}

torch::Tensor sage2_test_attention(
    torch::Tensor q_bmhd,
    torch::Tensor k_bmhd,
    torch::Tensor v_bmhd,
    bool causal) {
  TORCH_CHECK(q_bmhd.is_cuda() && k_bmhd.is_cuda() && v_bmhd.is_cuda(),
              "Sage2 Q/K/V inputs must be CUDA tensors");
  TORCH_CHECK(q_bmhd.scalar_type() == at::kBFloat16 &&
                  k_bmhd.scalar_type() == at::kBFloat16 &&
                  v_bmhd.scalar_type() == at::kBFloat16,
              "Sage2 Q/K/V inputs must be bfloat16");
  TORCH_CHECK(q_bmhd.dim() == 4 && k_bmhd.dim() == 4 && v_bmhd.dim() == 4,
              "Sage2 Q/K/V inputs must have BMHD rank 4");
  TORCH_CHECK(q_bmhd.is_contiguous() && k_bmhd.is_contiguous() &&
                  v_bmhd.is_contiguous(),
              "Sage2 Q/K/V inputs must be contiguous");
  TORCH_CHECK(q_bmhd.device() == k_bmhd.device() &&
                  q_bmhd.device() == v_bmhd.device(),
              "Sage2 Q/K/V inputs must be on the same CUDA device");
  TORCH_CHECK(q_bmhd.size(0) == k_bmhd.size(0) &&
                  q_bmhd.size(0) == v_bmhd.size(0) &&
                  q_bmhd.size(2) == k_bmhd.size(2) &&
                  q_bmhd.size(2) == v_bmhd.size(2) &&
                  q_bmhd.size(3) == k_bmhd.size(3) &&
                  q_bmhd.size(3) == v_bmhd.size(3) &&
                  k_bmhd.size(1) == v_bmhd.size(1),
              "Sage2 Q/K/V BMHD shapes are incompatible");

  auto output = torch::empty_like(q_bmhd);
  const auto stream = at::cuda::getCurrentCUDAStream(q_bmhd.get_device());
  cudaError_t err = run_sage2_fmha_packed_qkv(
      reinterpret_cast<const cutlass::bfloat16_t*>(q_bmhd.data_ptr()),
      reinterpret_cast<const cutlass::bfloat16_t*>(k_bmhd.data_ptr()),
      reinterpret_cast<const cutlass::bfloat16_t*>(v_bmhd.data_ptr()),
      reinterpret_cast<cutlass::bfloat16_t*>(output.data_ptr()),
      static_cast<int>(q_bmhd.size(0)),
      static_cast<int>(q_bmhd.size(1)),
      static_cast<int>(k_bmhd.size(1)),
      static_cast<int>(q_bmhd.size(2)),
      static_cast<int>(q_bmhd.size(3)),
      causal,
      0.0f,
      stream.stream());
  TORCH_CHECK(err == cudaSuccess,
              "Sage2 attention failed: ", cudaGetErrorString(err));
  return output;
}

}  // namespace omnidreams_singleview
