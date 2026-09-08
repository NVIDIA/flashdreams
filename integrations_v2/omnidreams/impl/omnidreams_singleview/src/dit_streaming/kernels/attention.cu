// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "attention.cuh"
#include "workspace.cuh"
#include "linear_utils.cuh"
#include "helper.h"
#include "kernel_forward.h"
#ifdef OMNIDREAMS_SINGLEVIEW_HAS_SPARGE
#include "sparge_attention_kernels.cuh"
#endif

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/numeric_types.h"
#include "cutlass/bfloat16.h"
#include "ops.cuh"
#include "dtype_utils.cuh"
#include "common/profile_config.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cudnn_frontend.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// Debug helper: check for NaN/Inf in a half buffer on device
// Uncomment `#define FP8_NAN_DEBUG 1` to enable
// #define FP8_NAN_DEBUG 1
#ifdef FP8_NAN_DEBUG
static __global__ void ca_nan_check_kernel(const half* data, int64_t n, int* found_nan, int* found_inf) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = __half2float(data[idx]);
  if (isnan(v)) atomicAdd(found_nan, 1);
  if (isinf(v)) atomicAdd(found_inf, 1);
}
static bool ca_check_nan_inf(const cutlass::half_t* buf, int64_t count, const char* label, cudaStream_t stream) {
  int h_nan = 0, h_inf = 0;
  int *d_nan, *d_inf;
  cudaMallocAsync(&d_nan, sizeof(int), stream);
  cudaMallocAsync(&d_inf, sizeof(int), stream);
  cudaMemsetAsync(d_nan, 0, sizeof(int), stream);
  cudaMemsetAsync(d_inf, 0, sizeof(int), stream);
  int threads = 256;
  int blocks = (int)((count + threads - 1) / threads);
  ca_nan_check_kernel<<<blocks, threads, 0, stream>>>(
    reinterpret_cast<const half*>(buf), count, d_nan, d_inf);
  cudaMemcpyAsync(&h_nan, d_nan, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&h_inf, d_inf, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFreeAsync(d_nan, stream);
  cudaFreeAsync(d_inf, stream);
  if (h_nan > 0 || h_inf > 0) {
    printf("[NAN_DEBUG] %s: %d NaN, %d Inf out of %lld elements\n", label, h_nan, h_inf, (long long)count);
    return true;
  }
  return false;
}
#define CA_NAN_CHECK(buf, count, label) ca_check_nan_inf(buf, count, label, stream)
#else
#define CA_NAN_CHECK(buf, count, label) false
#endif

namespace fe = cudnn_frontend;

namespace omnidreams_singleview {

// ============================================================================
// Optional fine-grained profiling helpers (level >= 3)
// ============================================================================

static inline bool wan_profile_detail_enabled(long long* out_call_idx) {
  int prof_lvl = g_wan_profile_level.load(std::memory_order_relaxed);
  if (prof_lvl < 3) return false;

  int print_every = g_wan_profile_print_every.load(std::memory_order_relaxed);
  if (print_every < 1) print_every = 1;

  // g_wan_profile_call_idx is incremented in the transformer block before calling attention.
  // We want the "current" call index, i.e. the last value returned by fetch_add.
  long long cur = g_wan_profile_call_idx.load(std::memory_order_relaxed);
  long long call_idx = (cur > 0) ? (cur - 1) : 0;
  if ((call_idx % print_every) != 0) return false;

  if (out_call_idx) *out_call_idx = call_idx;
  return true;
}

// ============================================================================
// Cosmos FP8 cuDNN SDPA engine/layout selection
// ============================================================================

namespace {

std::string cosmos_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool cosmos_env_is_set(const char* value) {
  return value && value[0] != '\0';
}

bool cosmos_valid_fp8_sdpa_layout(const std::string& layout) {
  return layout == "bmhd" || layout == "bhmd";
}

bool cosmos_valid_fp8_sdpa_heuristics(const std::string& heuristics) {
  return heuristics == "a" ||
         heuristics == "b" ||
         heuristics == "fallback" ||
         heuristics == "a_b" ||
         heuristics == "a_b_fallback" ||
         heuristics == "b_fallback";
}

bool cosmos_valid_fp8_sdpa_plan(const std::string& plan) {
  return plan == "heuristic" || plan == "all" || plan == "autotune";
}

std::vector<fe::HeurMode_t> cosmos_fp8_sdpa_heur_modes(const std::string& heuristics) {
  if (heuristics == "b") return {fe::HeurMode_t::B};
  if (heuristics == "fallback") return {fe::HeurMode_t::FALLBACK};
  if (heuristics == "a_b") return {fe::HeurMode_t::A, fe::HeurMode_t::B};
  if (heuristics == "a_b_fallback") {
    return {fe::HeurMode_t::A, fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK};
  }
  if (heuristics == "b_fallback") {
    return {fe::HeurMode_t::B, fe::HeurMode_t::FALLBACK};
  }
  return {fe::HeurMode_t::A};
}

fe::BuildPlanPolicy_t cosmos_fp8_sdpa_build_policy(const std::string& plan) {
  return plan == "all" || plan == "autotune"
      ? fe::BuildPlanPolicy_t::ALL
      : fe::BuildPlanPolicy_t::HEURISTICS_CHOICE;
}

}  // namespace

CosmosFp8SdpaSelection select_cosmos_fp8_sdpa(
    int B, int Mq, int Mk,
    int H, int D) {
  (void)B;
  (void)Mq;
  (void)Mk;
  (void)H;
  (void)D;

  CosmosFp8SdpaSelection selection;
  const char* preset_env = std::getenv("OMNIDREAMS_DIT_FP8_SDPA_PRESET");
  selection.preset = cosmos_env_is_set(preset_env) ? cosmos_lower(preset_env) : "540p";

  const char* layout_env = std::getenv("OMNIDREAMS_DIT_FP8_SDPA_LAYOUT");
  const char* heur_env = std::getenv("OMNIDREAMS_DIT_FP8_SDPA_HEUR");
  const char* plan_env = std::getenv("OMNIDREAMS_DIT_FP8_SDPA_PLAN");

  if (cosmos_env_is_set(layout_env)) {
    std::string requested = cosmos_lower(layout_env);
    if (cosmos_valid_fp8_sdpa_layout(requested)) {
      selection.layout = requested;
      selection.layout_env_override = true;
      selection.reason = "OMNIDREAMS_DIT_FP8_SDPA_LAYOUT_override";
    } else {
      selection.reason = "ignored_invalid_OMNIDREAMS_DIT_FP8_SDPA_LAYOUT";
    }
  }

  if (cosmos_env_is_set(heur_env)) {
    std::string requested = cosmos_lower(heur_env);
    if (cosmos_valid_fp8_sdpa_heuristics(requested)) {
      selection.heuristics = requested;
      selection.heuristics_env_override = true;
      if (selection.reason.empty()) {
        selection.reason = "OMNIDREAMS_DIT_FP8_SDPA_HEUR_override";
      }
    } else if (selection.reason.empty() || selection.reason == "540p_preset") {
      selection.reason = "ignored_invalid_OMNIDREAMS_DIT_FP8_SDPA_HEUR";
    }
  }

  if (cosmos_env_is_set(plan_env)) {
    std::string requested = cosmos_lower(plan_env);
    if (cosmos_valid_fp8_sdpa_plan(requested)) {
      selection.plan = requested;
      selection.plan_env_override = true;
      if (selection.reason.empty()) {
        selection.reason = "OMNIDREAMS_DIT_FP8_SDPA_PLAN_override";
      }
    } else if (selection.reason.empty() || selection.reason == "540p_preset") {
      selection.reason = "ignored_invalid_OMNIDREAMS_DIT_FP8_SDPA_PLAN";
    }
  }

  if (selection.layout.empty() || selection.heuristics.empty() || selection.plan.empty()) {
    if (selection.preset == "legacy" || selection.preset == "env") {
      if (selection.layout.empty()) selection.layout = "bmhd";
      if (selection.heuristics.empty()) selection.heuristics = "a";
      if (selection.plan.empty()) selection.plan = "heuristic";
      if (selection.reason.empty()) {
        selection.reason = selection.preset == "env"
            ? "env_missing_fields_legacy_fallback"
            : "legacy";
      }
    } else {
      // 540p preset, retained until a measured autotune winner beats it:
      // BMHD input avoids extra transposes; A+B heuristics with all successful
      // plans was the 2026-05-06 540p E2E winner in
      // profiles/cosmos_fp8_attention/autotune_sdpa_540p_20260506_184914_summary.json.
      if (selection.layout.empty()) selection.layout = "bmhd";
      if (selection.heuristics.empty()) selection.heuristics = "a_b";
      if (selection.plan.empty()) selection.plan = "all";
      if (selection.reason.empty()) {
        selection.reason = selection.preset == "540p"
            ? "540p_preset"
            : "unknown_preset_540p_fallback";
      }
    }
  }

  return selection;
}

// ============================================================================
// Attention Backend Selection
// ============================================================================

namespace {
    AttnBackend g_attention_backend = AttnBackend::CUTLASS_FLASH;
}

void set_attention_backend(AttnBackend backend) {
    g_attention_backend = backend;
}

AttnBackend get_attention_backend() {
    return g_attention_backend;
}

static GemmBackend g_gemm_backend = GemmBackend::FP16;
void set_gemm_backend(GemmBackend b) { g_gemm_backend = b; }
GemmBackend get_gemm_backend() { return g_gemm_backend; }

// ============================================================================
// Backend Implementations
// ============================================================================

// RowMajor fused variant: RMSNorm+pack BMHK and apply RoPE for Q,K in one pass
__global__ void rmsnorm_pack_bmhk_rope_from_row_kernel(
    const half* __restrict__ Drow, int M, int K,
    int H, int Dh,
    int M_rope,  // RoPE table length (per-sample sequence length, i.e. Mq)
    const cutlass::half_t* __restrict__ gamma_q, const cutlass::half_t* __restrict__ gamma_k, float eps,
    const float* __restrict__ cos_tbl, // [M, D] row-major
    const float* __restrict__ sin_tbl, // [M, D] row-major
    cutlass::half_t* __restrict__ Q_bmhk,
    cutlass::half_t* __restrict__ K_bmhk,
    cutlass::half_t* __restrict__ V_bmhk) {
  int row = blockIdx.x; if (row >= M) return;
  int tid = threadIdx.x;
  extern __shared__ float ssum[];
  float* ssum_q = ssum;
  float* ssum_k = ssum + blockDim.x;
  // base pointer for this row
  const half* row_ptr = Drow + size_t(row) * (3 * K);
  // Accumulate sums for RMSNorm of Q and K
  float sq = 0.f, sk = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float qv = __half2float(row_ptr[j]);
    float kv = __half2float(row_ptr[K + j]);
    sq += qv * qv;
    sk += kv * kv;
  }
  ssum_q[tid] = sq;
  ssum_k[tid] = sk;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) {
      ssum_q[tid] += ssum_q[tid + off];
      ssum_k[tid] += ssum_k[tid + off];
    }
    __syncthreads();
  }
  float inv_q = rsqrtf(ssum_q[0] / float(K) + eps);
  float inv_k = rsqrtf(ssum_k[0] / float(K) + eps);

  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;

  // Pointers to RoPE rows for this sequence position.
  // For batched inputs, tokens are flattened as [B*Mq, ...] and share a single RoPE table [Mq, Dh].
  // So we index RoPE by m = row % M_rope, where M_rope == Mq.
  int rope_m = (cos_tbl && sin_tbl) ? (row % M_rope) : 0;
  const float* cos_row = cos_tbl ? (cos_tbl + size_t(rope_m) * Dh) : nullptr;
  const float* sin_row = sin_tbl ? (sin_tbl + size_t(rope_m) * Dh) : nullptr;

  // Write BMHK with RoPE applied to Q and K. Process pairs (even, odd) per head.
  // Iterate over K with step 2 per thread to avoid write races between pair elements.
  for (int j = tid * 2; j + 1 < K; j += blockDim.x * 2) {
    int p = j % Dh; // position within head
    // Load and normalize Q pair
    float q_even = __half2float(row_ptr[j]);
    float q_odd  = __half2float(row_ptr[j + 1]);
    float q_ne = (q_even * inv_q) * to_f32(gamma_q[j]);
    float q_no = (q_odd  * inv_q) * to_f32(gamma_q[j + 1]);
    // Load and normalize K pair
    float k_even = __half2float(row_ptr[K + j]);
    float k_odd  = __half2float(row_ptr[K + j + 1]);
    float k_ne = (k_even * inv_k) * to_f32(gamma_k[j]);
    float k_no = (k_odd  * inv_k) * to_f32(gamma_k[j + 1]);

    if (cos_row && sin_row) {
      // Apply RoPE rotation using per-dimension cos/sin
      float c = cos_row[p];
      float s = sin_row[p + 1];
      // Q rotation
      float q_rot_e = q_ne * c - q_no * s;
      float q_rot_o = q_ne * s + q_no * c;
      q_ne = q_rot_e; q_no = q_rot_o;
      // K rotation
      float k_rot_e = k_ne * c - k_no * s;
      float k_rot_o = k_ne * s + k_no * c;
      k_ne = k_rot_e; k_no = k_rot_o;
    }

    size_t base = size_t(row) * K;
    // Write Q
    Q_bmhk[base + j]     = to_f16(q_ne);
    Q_bmhk[base + j + 1] = to_f16(q_no);
    // Write K
    K_bmhk[base + j]     = to_f16(k_ne);
    K_bmhk[base + j + 1] = to_f16(k_no);
    // Write V (no RMSNorm, no RoPE)
    float v0 = __half2float(row_ptr[2 * K + j]);
    float v1 = __half2float(row_ptr[2 * K + j + 1]);
    V_bmhk[base + j]     = to_f16(v0);
    V_bmhk[base + j + 1] = to_f16(v1);
  }

  // If Dh is odd (unlikely), handle the last lane without RoPE pairwise rotation
  if ((Dh & 1) && (tid == 0)) {
    // Process indices j where (j % Dh) == Dh-1
    for (int h = 0; h < H; ++h) {
      int j = h * Dh + (Dh - 1);
      if (j < K) {
        float qv = __half2float(row_ptr[j]);
        float kv = __half2float(row_ptr[K + j]);
        float vv = __half2float(row_ptr[2 * K + j]);
        float qn = (qv * inv_q) * to_f32(gamma_q[j]);
        float kn = (kv * inv_k) * to_f32(gamma_k[j]);
        Q_bmhk[size_t(row) * K + j] = to_f16(qn);
        K_bmhk[size_t(row) * K + j] = to_f16(kn);
        V_bmhk[size_t(row) * K + j] = to_f16(vv);
      }
    }
  }
}


// Pack one matrix with RMSNorm (half input), add bias and gamma
__global__ void rmsnorm_pack_single_from_half_kernel(
    const cutlass::half_t* __restrict__ Dcol_h, int ld, int rows, int K,
    int H, int Dh,
    const cutlass::half_t* __restrict__ gamma,
    float eps,
    cutlass::half_t* __restrict__ Out_bmhk) {
  int row = blockIdx.x; if (row >= rows) return;
  int tid = threadIdx.x;
  extern __shared__ float ssum[];
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  float acc = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = to_f32(Dcol_h[row + j * ld]);
    acc += v * v;
  }
  ssum[tid] = acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) { if (tid < off) ssum[tid] += ssum[tid + off]; __syncthreads(); }
  float inv = rsqrtf(ssum[0] / float(K) + eps);
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;
  size_t row_base = size_t(row) * K;
  for (int j = tid; j < K; j += blockDim.x) {
    float vn = (to_f32(Dcol_h[row + j * ld])) * inv * to_f32(gamma[j]);
    Out_bmhk[row_base + j] = to_f16(vn);
  }
}

// Pack K with RMSNorm (gamma_k) and V as-is from concatenated [rows, 2K] half into BMHK half
__global__ void rmsnorm_pack_kv_from_half_kernel(
    const cutlass::half_t* __restrict__ Dcol_h, int ld, int rows, int K,
    int H, int Dh,
    const cutlass::half_t* __restrict__ gamma_k,
    float eps,
    cutlass::half_t* __restrict__ K_bmhk,
    cutlass::half_t* __restrict__ V_bmhk) {
  int row = blockIdx.x; if (row >= rows) return;
  int tid = threadIdx.x;
  extern __shared__ float ssum[];
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  float acc = 0.f;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = to_f32(Dcol_h[row + j * ld]);
    acc += v * v;
  }
  ssum[tid] = acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) { if (tid < off) ssum[tid] += ssum[tid + off]; __syncthreads(); }
  float inv = rsqrtf(ssum[0] / float(K) + eps);
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;
  size_t row_base = size_t(row) * K;
  for (int j = tid; j < K; j += blockDim.x) {
    float kn = (to_f32(Dcol_h[row + j * ld])) * inv * to_f32(gamma_k[j]);
    K_bmhk[row_base + j] = to_f16(kn);
    float vv = to_f32(Dcol_h[row + (K + j) * ld]);
    V_bmhk[row_base + j] = to_f16(vv);
  }
}

// RowMajor variant: Pack one matrix with RMSNorm (row-major input)
__global__ void rmsnorm_pack_single_from_row_kernel(
    const half* __restrict__ Drow, int rows, int K,
    int H, int Dh,
    const cutlass::half_t* __restrict__ gamma,
    float eps,
    cutlass::half_t* __restrict__ Out_bmhk) {
  int row = blockIdx.x; if (row >= rows) return;
  int tid = threadIdx.x;
  extern __shared__ float ssum[];
  float acc = 0.f;
  const half* row_ptr = Drow + size_t(row) * K;
  for (int j = tid; j < K; j += blockDim.x) {
    float v = __half2float(row_ptr[j]);
    acc += v * v;
  }
  ssum[tid] = acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) { if (tid < off) ssum[tid] += ssum[tid + off]; __syncthreads(); }
  float inv = rsqrtf(ssum[0] / float(K) + eps);
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;
  size_t row_base = size_t(row) * K;
  for (int j = tid; j < K; j += blockDim.x) {
    float vn = (__half2float(row_ptr[j])) * inv * to_f32(gamma[j]);
    Out_bmhk[row_base + j] = to_f16(vn);
  }
}

// RowMajor variant: Pack K with RMSNorm (gamma_k) and V as-is from concatenated [rows, 2K] row-major into BMHK half
__global__ void rmsnorm_pack_kv_from_row_kernel(
    const half* __restrict__ Drow, int rows, int K,
    int H, int Dh,
    const cutlass::half_t* __restrict__ gamma_k,
    float eps,
    cutlass::half_t* __restrict__ K_bmhk,
    cutlass::half_t* __restrict__ V_bmhk) {
  int row = blockIdx.x; if (row >= rows) return;
  int tid = threadIdx.x;
  extern __shared__ float ssum[];
  float acc = 0.f;
  const half* row_ptr = Drow + size_t(row) * (2 * K);
  for (int j = tid; j < K; j += blockDim.x) {
    float v = __half2float(row_ptr[j]);
    acc += v * v;
  }
  ssum[tid] = acc;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) { if (tid < off) ssum[tid] += ssum[tid + off]; __syncthreads(); }
  float inv = rsqrtf(ssum[0] / float(K) + eps);
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;
  size_t row_base = size_t(row) * K;
  for (int j = tid; j < K; j += blockDim.x) {
    float kn = (__half2float(row_ptr[j])) * inv * to_f32(gamma_k[j]);
    K_bmhk[row_base + j] = to_f16(kn);
    float vv = __half2float(row_ptr[K + j]);
    V_bmhk[row_base + j] = to_f16(vv);
  }
}

// Apply RoPE in-place on Q and K packed as BMHK (half precision)
__global__ void apply_rope_bmhk_kernel(
    cutlass::half_t* __restrict__ Q_bmhk,
    cutlass::half_t* __restrict__ K_bmhk,
    int M, int H, int D,
    const float* __restrict__ cos_tbl, // [M, D] row-major
    const float* __restrict__ sin_tbl  // [M, D] row-major
    ) {
  int i = blockIdx.y; // sequence position
  int h = blockIdx.x; // head
  if (i >= M || h >= H) return;
  int t = threadIdx.x;
  cutlass::NumericConverter<float, cutlass::half_t> to_f32;
  cutlass::NumericConverter<cutlass::half_t, float> to_f16;
  size_t base = size_t(i) * (H * D) + size_t(h) * D;
  const float* cos_row = cos_tbl + size_t(i) * D;
  const float* sin_row = sin_tbl + size_t(i) * D;
  // process pairs (even, odd)
  for (int p = t * 2; p + 1 < D; p += blockDim.x * 2) {
    float c = cos_row[p];
    float s = sin_row[p + 1];
    // Q
    float q_even = to_f32(Q_bmhk[base + p]);
    float q_odd  = to_f32(Q_bmhk[base + p + 1]);
    float q_ne = q_even * c - q_odd * s;
    float q_no = q_even * s + q_odd * c;
    Q_bmhk[base + p] = to_f16(q_ne);
    Q_bmhk[base + p + 1] = to_f16(q_no);
    // K
    float k_even = to_f32(K_bmhk[base + p]);
    float k_odd  = to_f32(K_bmhk[base + p + 1]);
    float k_ne = k_even * c - k_odd * s;
    float k_no = k_even * s + k_odd * c;
    K_bmhk[base + p] = to_f16(k_ne);
    K_bmhk[base + p + 1] = to_f16(k_no);
  }
}

// CUTLASS Flash Attention implementation
static cudaError_t run_self_attention_cutlass(const AttentionDeviceParamsT<cutlass::half_t>& p, cudaStream_t stream) {
  int B = p.B;
  int Mq = p.Mq;
  if (Mq <= 0) { return cudaErrorInvalidValue; }
  int M = B * Mq;
  int K = p.K, H = p.H, D = p.D;

  // Optional fine-grained profiling (GPU time via CUDA events).
  long long prof_call_idx = 0;
  bool prof_detail = wan_profile_detail_enabled(&prof_call_idx);
  enum {
    EV_START = 0,
    EV_AFTER_QKV_PROJ,
    EV_AFTER_PACK_ROPE,
    EV_AFTER_FMHA,
    EV_AFTER_OUT,
    EV_COUNT
  };
  cudaEvent_t ev[EV_COUNT];
  auto rec = [&](int idx) {
    if (prof_detail) cudaEventRecord(ev[idx], stream);
  };
  if (prof_detail) {
    for (int i = 0; i < EV_COUNT; ++i) cudaEventCreate(&ev[i]);
    rec(EV_START);
  }

  // Use workspace (always provided)
  cutlass::half_t* qkv_row = p.workspace->cutlass_flash.sa_qkv_row;
  cutlass::half_t* dQh_bmhk = p.workspace->cutlass_flash.sa_q_bmhk;
  cutlass::half_t* dKh_bmhk = p.workspace->cutlass_flash.sa_k_bmhk;
  cutlass::half_t* dVh_bmhk = p.workspace->cutlass_flash.sa_v_bmhk;
  cutlass::half_t* dOh_bmhk = p.workspace->cutlass_flash.sa_o_bmhk;  // Reuses sa_q_bmhk

  // 1) QKV projection via cutlass_linear_layer (weights pre-transposed in Python to [K,3K])
  // Expect hidden_states to be RowMajor [M,K] already (avoid extra transpose)
  // RowMajor GEMM with stride-trick bias to compute QKV into RowMajor buffer
  {
    auto* fp8_scratch = reinterpret_cast<cutlass::half_t*>(dQh_bmhk); // scratch for potential FP8 path
    cudaError_t err = apply_linear_row<cutlass::half_t>(
        reinterpret_cast<const cutlass::half_t*>(p.hidden_states),
        reinterpret_cast<const cutlass::half_t*>(p.w_qkv),
        reinterpret_cast<const cutlass::half_t*>(p.b_qkv),
        reinterpret_cast<cutlass::half_t*>(qkv_row),
        M, K, 3 * K, stream,
        fp8_scratch);
    if (err != cudaSuccess) { return err; }
  }
  rec(EV_AFTER_QKV_PROJ);

  // Fused RMSNorm+pack+RoPE
  rmsnorm_pack_bmhk_rope_from_row_kernel<<<M, 256, 2*256*sizeof(float), stream>>>(
    reinterpret_cast<const half*>(qkv_row), M, K, H, D, Mq,
    p.norm_q_gamma, p.norm_k_gamma, 1e-6f,
    p.rotary_cos, p.rotary_sin,
    dQh_bmhk, dKh_bmhk, dVh_bmhk);
  CUDA_CHECK(cudaGetLastError());
  rec(EV_AFTER_PACK_ROPE);

  // 3) FMHA using CUTLASS example kernel (half precision) directly on packed tensors
  using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128, false, false>;
  typename Attention::Params ap{};
  ap.query_ptr = dQh_bmhk; ap.key_ptr = dKh_bmhk; ap.value_ptr = dVh_bmhk;
  ap.logsumexp_ptr = nullptr; ap.output_accum_ptr = nullptr; ap.output_ptr = dOh_bmhk;
  ap.scale = 1.0f / sqrtf(float(D)); ap.num_heads = H; ap.num_batches = B; ap.head_dim = D; ap.head_dim_value = D; ap.num_queries = Mq; ap.num_keys = Mq;
  ap.q_strideH = D; ap.k_strideH = D; ap.v_strideH = D; ap.q_strideM = H * D; ap.k_strideM = H * D; ap.v_strideM = H * D; ap.q_strideB = ap.q_strideM * ap.num_queries; ap.k_strideB = ap.k_strideM * ap.num_keys; ap.v_strideB = ap.v_strideM * ap.num_keys; ap.o_strideM = ap.head_dim_value * ap.num_heads;
  ap.custom_mask_type = p.is_causal ? Attention::CausalFromTopLeft : Attention::NoCustomMask;
  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) { cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); }
  if (!Attention::check_supported(ap)) return cudaErrorNotSupported;
  kernel_fn<<<ap.getBlocksGrid(), ap.getThreadsGrid(), smem_bytes, stream>>>(ap);
  CUDA_CHECK(cudaGetLastError());
  rec(EV_AFTER_FMHA);

  // 4) Output projection via shared helper (consume BMHK as row-major [M,K])
  {
    auto* fp8_scratch = reinterpret_cast<cutlass::half_t*>(dQh_bmhk); // safe scratch
    cudaError_t err = apply_linear_row<cutlass::half_t>(
        reinterpret_cast<const cutlass::half_t*>(dOh_bmhk),
        reinterpret_cast<const cutlass::half_t*>(p.w_out),
        reinterpret_cast<const cutlass::half_t*>(p.b_out),
        reinterpret_cast<cutlass::half_t*>(p.out_after_linear),
        M, K, K, stream,
        fp8_scratch);
    if (err != cudaSuccess) { return err; }

    rec(EV_AFTER_OUT);
    if (prof_detail) {
      cudaEventSynchronize(ev[EV_AFTER_OUT]);
      auto ms = [&](int a, int b) -> float {
        float out = 0.f;
        cudaEventElapsedTime(&out, ev[a], ev[b]);
        return out;
      };
      float t_qkv  = ms(EV_START, EV_AFTER_QKV_PROJ);
      float t_pack = ms(EV_AFTER_QKV_PROJ, EV_AFTER_PACK_ROPE);
      float t_fmha = ms(EV_AFTER_PACK_ROPE, EV_AFTER_FMHA);
      float t_out  = ms(EV_AFTER_FMHA, EV_AFTER_OUT);
      float t_tot  = ms(EV_START, EV_AFTER_OUT);
      std::printf(
        "[sa_cutlass][call=%lld] M=%d K=%d H=%d D=%d | "
        "qkv=%.3f pack_rope=%.3f fmha=%.3f out=%.3f total=%.3f ms\n",
        prof_call_idx, M, K, H, D, t_qkv, t_pack, t_fmha, t_out, t_tot
      );
      for (int i = 0; i < EV_COUNT; ++i) cudaEventDestroy(ev[i]);
    }
  }
  return cudaSuccess;
}

// CUTLASS Flash Attention implementation
static cudaError_t run_cross_attention_cutlass(const AttentionDeviceParamsT<cutlass::half_t>& p, bool fuse_residual, cudaStream_t stream) {
  int B = p.B;
  int Mq = p.Mq;
  if (Mq <= 0) { return cudaErrorInvalidValue; }
  int M = B * Mq;
  int K = p.K, H = p.H, D = p.D;
  bool use_image_branch = (p.w_kv == nullptr) && (p.w_add_k != nullptr) && (p.w_add_v != nullptr);
  int Mk = use_image_branch ? (p.Mk_img ? p.Mk_img : Mq) : (p.Mk ? p.Mk : Mq);

  // Determine encoder batch size for KV projection (shared KV optimization for CFG)
  // If encoder_batch_size=0, use B (standard behavior). If encoder_batch_size=1 and B>1,
  // project KV once and broadcast to all batch items.
  int enc_B = (p.encoder_batch_size > 0) ? p.encoder_batch_size : B;
  bool shared_kv = (enc_B < B);
  int kv_rows = enc_B * Mk;

  cutlass::half_t* dQh_bmhk = p.workspace->cutlass_flash.ca_q_bmhk;
  cutlass::half_t* dKh_bmhk = p.workspace->cutlass_flash.ca_k_bmhk;
  cutlass::half_t* dVh_bmhk = p.workspace->cutlass_flash.ca_v_bmhk;
  cutlass::half_t* dOh_bmhk = p.workspace->cutlass_flash.ca_o_bmhk;  // Reuses ca_q_bmhk
  cutlass::half_t* dQrow = p.workspace->cutlass_flash.ca_q_row;       // Reuses portion of sa_qkv_row
  cutlass::half_t* dKVrow = p.workspace->cutlass_flash.ca_kv_row;

  // 1a) Q projection with row-major linear (stride-trick bias)
  {
    cudaError_t err = cutlass_linear_layer_rrr(
        reinterpret_cast<const half*>(p.hidden_states),
        reinterpret_cast<const half*>(p.w_q),
        reinterpret_cast<const half*>(p.b_q),
        reinterpret_cast<half*>(dQrow),
        M, K, K, stream);
    if (err != cudaSuccess) { return err; }
    // Pack Q directly from row-major
    rmsnorm_pack_single_from_row_kernel<<<M, 256, 256*sizeof(float), stream>>>(
      reinterpret_cast<const half*>(dQrow), M, K, H, D,
      p.norm_q_gamma, 1.0e-6f,
      dQh_bmhk);
    CUDA_CHECK(cudaGetLastError());
  }

  // 1b) KV projection with row-major linear (stride-trick bias)
  // When shared_kv=true, project only enc_B*Mk tokens instead of B*Mk.
  if (!use_image_branch) {
    cudaError_t err = cutlass_linear_layer_rrr(
        reinterpret_cast<const half*>(p.encoder_hidden_states),
        reinterpret_cast<const half*>(p.w_kv),
        reinterpret_cast<const half*>(p.b_kv),
        reinterpret_cast<half*>(dKVrow),
        kv_rows, K, 2 * K, stream);
    if (err != cudaSuccess) { return err; }
    // Pack K and V directly from row-major
    rmsnorm_pack_kv_from_row_kernel<<<kv_rows, 256, 256*sizeof(float), stream>>>(
      reinterpret_cast<const half*>(dKVrow), kv_rows, K, H, D,
      p.norm_k_gamma, 1.0e-6f,
      dKh_bmhk, dVh_bmhk);
    CUDA_CHECK(cudaGetLastError());
  } else {
    // Image added-KV path: separate K/V projections
    CUDA_CHECK(p.added_kv_proj_dim > 0 ? cudaSuccess : cudaErrorInvalidValue);
    // IMPORTANT: dKVrow is a [kv_rows, 2K] row-major buffer. We must write K and V
    // into it with row stride = 2K, otherwise rows get scrambled.
    cudaError_t err_k = cutlass_linear_layer_rrr_strided(
        reinterpret_cast<const half*>(p.encoder_hidden_states),
        reinterpret_cast<const half*>(p.w_add_k),
        reinterpret_cast<const half*>(p.b_add_k),
        reinterpret_cast<half*>(dKVrow),
        kv_rows, p.added_kv_proj_dim, K, 2 * K, stream);
    if (err_k != cudaSuccess) { return err_k; }
    cudaError_t err_v = cutlass_linear_layer_rrr_strided(
        reinterpret_cast<const half*>(p.encoder_hidden_states),
        reinterpret_cast<const half*>(p.w_add_v),
        reinterpret_cast<const half*>(p.b_add_v),
        reinterpret_cast<half*>(dKVrow) + K,
        kv_rows, p.added_kv_proj_dim, K, 2 * K, stream);
    if (err_v != cudaSuccess) { return err_v; }
    rmsnorm_pack_kv_from_row_kernel<<<kv_rows, 256, 256*sizeof(float), stream>>>(
      reinterpret_cast<const half*>(dKVrow), kv_rows, K, H, D,
      p.norm_added_k_gamma, 1.0e-6f,
      dKh_bmhk, dVh_bmhk);
    CUDA_CHECK(cudaGetLastError());
  }

  // 2) Pack Q, K, V into BMHK half (already packed above)

  // 3) FMHA
  using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128, false, false>;
  typename Attention::Params ap{};
  ap.query_ptr = dQh_bmhk; ap.key_ptr = dKh_bmhk; ap.value_ptr = dVh_bmhk;
  ap.logsumexp_ptr = nullptr; ap.output_accum_ptr = nullptr; ap.output_ptr = dOh_bmhk;
  ap.scale = 1.0f / sqrtf(float(D)); ap.num_heads = H; ap.num_batches = B; ap.head_dim = D; ap.head_dim_value = D; ap.num_queries = Mq; ap.num_keys = Mk;
  ap.q_strideH = D; ap.k_strideH = D; ap.v_strideH = D; ap.q_strideM = H * D; ap.k_strideM = H * D; ap.v_strideM = H * D; ap.q_strideB = ap.q_strideM * ap.num_queries;
  // When shared_kv=true, set K/V batch stride to 0 to broadcast the single KV set to all batch items
  ap.k_strideB = shared_kv ? 0 : (ap.k_strideM * ap.num_keys);
  ap.v_strideB = shared_kv ? 0 : (ap.v_strideM * ap.num_keys);
  ap.o_strideM = ap.head_dim_value * ap.num_heads;
  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) { cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); }
  if (!Attention::check_supported(ap)) return cudaErrorNotSupported;
  kernel_fn<<<ap.getBlocksGrid(), ap.getThreadsGrid(), smem_bytes, stream>>>(ap);
  CUDA_CHECK(cudaGetLastError());

  // 4) Output projection (optionally fused with residual)
  // Flash attention outputs in row-major [M, K] format already (o_strideM = H*D = K)
  cudaError_t ca_err = cudaSuccess;
  if (fuse_residual) {
    ca_err = cutlass_linear_layer_rrr_fused_residual(
        reinterpret_cast<const half*>(dOh_bmhk),
        reinterpret_cast<const half*>(p.w_out),
        reinterpret_cast<const half*>(p.b_out),
        reinterpret_cast<half*>(p.out_after_linear),  // residual inout
        M, K, K, stream);
  } else {
    ca_err = cutlass_linear_layer_rrr(
        reinterpret_cast<const half*>(dOh_bmhk),
        reinterpret_cast<const half*>(p.w_out),
        reinterpret_cast<const half*>(p.b_out),
        reinterpret_cast<half*>(p.out_after_linear),
        M, K, K, stream);
  }
  if (ca_err != cudaSuccess) { return ca_err; }
  return cudaSuccess;
}

// ============================================================================
// I2V fused cross-attention (CUTLASS backend): share Q across text + image branches
// ============================================================================

__global__ void add_inplace_half_kernel_1d(half* __restrict__ dst, const half* __restrict__ src, int64_t n) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    dst[i] = __hadd(dst[i], src[i]);
  }
}

static cudaError_t run_cross_attention_i2v_cutlass(
    const CrossAttentionI2VParamsT<cutlass::half_t>& p,
    bool fuse_residual,
    cudaStream_t stream) {
  // CUTLASS fused I2V path currently supports only B=1.
  if (p.B != 1) return cudaErrorNotSupported;

  int M = p.M, K = p.K, H = p.H, D = p.D;
  int Mk_text = p.Mk_text ? p.Mk_text : p.M;

  // Image branch enabled only when all required pointers are present.
  bool do_img = (p.Mk_img > 0) &&
                (p.encoder_hidden_states_img != nullptr) &&
                (p.w_add_k != nullptr) && (p.w_add_v != nullptr) &&
                (p.norm_added_k_gamma != nullptr);
  int Mk_img = do_img ? p.Mk_img : 0;
  int Mk_total = Mk_text + Mk_img;  // fused KV length when image branch is present

  // Optional fine-grained profiling (GPU time via CUDA events).
  long long prof_call_idx = 0;
  bool prof_detail = wan_profile_detail_enabled(&prof_call_idx);
  enum {
    EV_START = 0,
    EV_AFTER_Q_PROJ,
    EV_AFTER_Q_PACK,
    EV_AFTER_TEXT_KV_PROJ,
    EV_AFTER_TEXT_KV_PACK,
    EV_AFTER_IMG_K_PROJ,
    EV_AFTER_IMG_V_PROJ,
    EV_AFTER_IMG_KV_PACK,
    EV_AFTER_FMHA,
    EV_AFTER_OUT_PROJ,
    EV_COUNT
  };
  cudaEvent_t ev[EV_COUNT];
  auto rec = [&](int idx) {
    if (prof_detail) cudaEventRecord(ev[idx], stream);
  };
  if (prof_detail) {
    for (int i = 0; i < EV_COUNT; ++i) cudaEventCreate(&ev[i]);
    rec(EV_START);
  }

  // Workspace pointers
  cutlass::half_t* dQh_bmhk = p.workspace->cutlass_flash.ca_q_bmhk;   // [M,K]
  cutlass::half_t* dKh_bmhk = p.workspace->cutlass_flash.ca_k_bmhk;   // [Mk_max,K]
  cutlass::half_t* dVh_bmhk = p.workspace->cutlass_flash.ca_v_bmhk;   // [Mk_max,K]
  cutlass::half_t* dQrow    = p.workspace->cutlass_flash.ca_q_row;    // [M,K]
  cutlass::half_t* dKVrow   = p.workspace->cutlass_flash.ca_kv_row;   // [Mk_max,2K]

  // Accumulator for attention outputs (row-major [M,K])
  // NOTE: Reuse block scratch (previously out_text).
  half* out_acc = reinterpret_cast<half*>(p.workspace->scratch_mk_b);

  // 1) Q projection (once) + RMSNorm/pack
  {
    cudaError_t err = cutlass_linear_layer_rrr(
        reinterpret_cast<const half*>(p.hidden_states),
        reinterpret_cast<const half*>(p.w_q),
        reinterpret_cast<const half*>(p.b_q),
        reinterpret_cast<half*>(dQrow),
        M, K, K, stream);
    if (err != cudaSuccess) { return err; }
    rec(EV_AFTER_Q_PROJ);

    rmsnorm_pack_single_from_row_kernel<<<M, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const half*>(dQrow), M, K, H, D,
        p.norm_q_gamma, 1.0e-6f,
        dQh_bmhk);
    CUDA_CHECK(cudaGetLastError());
    rec(EV_AFTER_Q_PACK);
  }

  // Helper: run FMHA for a given KV length using the shared packed Q.
  auto run_fmha = [&](int Mk, cutlass::half_t* out_ptr) -> cudaError_t {
    using Attention = AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128, false, false>;
    typename Attention::Params ap{};
    ap.query_ptr = dQh_bmhk; ap.key_ptr = dKh_bmhk; ap.value_ptr = dVh_bmhk;
    ap.logsumexp_ptr = nullptr; ap.output_accum_ptr = nullptr; ap.output_ptr = out_ptr;
    ap.scale = 1.0f / sqrtf(float(D));
    ap.num_heads = H; ap.num_batches = 1;
    ap.head_dim = D; ap.head_dim_value = D;
    ap.num_queries = M; ap.num_keys = Mk;
    ap.q_strideH = D; ap.k_strideH = D; ap.v_strideH = D;
    ap.q_strideM = H * D; ap.k_strideM = H * D; ap.v_strideM = H * D;
    ap.q_strideB = ap.q_strideM * ap.num_queries;
    ap.k_strideB = ap.k_strideM * ap.num_keys;
    ap.v_strideB = ap.v_strideM * ap.num_keys;
    ap.o_strideM = ap.head_dim_value * ap.num_heads;
    constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
    int smem_bytes = sizeof(typename Attention::SharedStorage);
    if (smem_bytes > 0xc000) { cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); }
    if (!Attention::check_supported(ap)) return cudaErrorNotSupported;
    kernel_fn<<<ap.getBlocksGrid(), ap.getThreadsGrid(), smem_bytes, stream>>>(ap);
    CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
  };

  // 2) Text KV -> pack
  {
    cudaError_t err = cutlass_linear_layer_rrr(
        reinterpret_cast<const half*>(p.encoder_hidden_states_text),
        reinterpret_cast<const half*>(p.w_kv),
        reinterpret_cast<const half*>(p.b_kv),
        reinterpret_cast<half*>(dKVrow),
        Mk_text, K, 2 * K, stream);
    if (err != cudaSuccess) { return err; }
    rec(EV_AFTER_TEXT_KV_PROJ);

    rmsnorm_pack_kv_from_row_kernel<<<Mk_text, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const half*>(dKVrow), Mk_text, K, H, D,
        p.norm_k_gamma, 1.0e-6f,
        dKh_bmhk, dVh_bmhk);
    CUDA_CHECK(cudaGetLastError());
    rec(EV_AFTER_TEXT_KV_PACK);
  }

  // 3) Image KV -> pack into the contiguous tail after text, then run a single FMHA over [text; image]
  if (do_img) {
    CUDA_CHECK(p.added_kv_proj_dim > 0 ? cudaSuccess : cudaErrorInvalidValue);

    // IMPORTANT: dKVrow is [Mk_img, 2K] row-major. Write K and V with row stride=2K.
    half* kv_row_img = reinterpret_cast<half*>(dKVrow + size_t(Mk_text) * 2 * K);
    cudaError_t err_k = cutlass_linear_layer_rrr_strided(
        reinterpret_cast<const half*>(p.encoder_hidden_states_img),
        reinterpret_cast<const half*>(p.w_add_k),
        reinterpret_cast<const half*>(p.b_add_k),
        kv_row_img,
        Mk_img, p.added_kv_proj_dim, K, 2 * K, stream);
    if (err_k != cudaSuccess) { return err_k; }
    rec(EV_AFTER_IMG_K_PROJ);

    cudaError_t err_v = cutlass_linear_layer_rrr_strided(
        reinterpret_cast<const half*>(p.encoder_hidden_states_img),
        reinterpret_cast<const half*>(p.w_add_v),
        reinterpret_cast<const half*>(p.b_add_v),
        kv_row_img + K,
        Mk_img, p.added_kv_proj_dim, K, 2 * K, stream);
    if (err_v != cudaSuccess) { return err_v; }
    rec(EV_AFTER_IMG_V_PROJ);

    // Pack image KV into the tail of dKh_bmhk / dVh_bmhk
    cutlass::half_t* dKh_img = dKh_bmhk + size_t(Mk_text) * H * D;
    cutlass::half_t* dVh_img = dVh_bmhk + size_t(Mk_text) * H * D;
    rmsnorm_pack_kv_from_row_kernel<<<Mk_img, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const half*>(kv_row_img), Mk_img, K, H, D,
        p.norm_added_k_gamma, 1.0e-6f,
        dKh_img, dVh_img);
    CUDA_CHECK(cudaGetLastError());
    rec(EV_AFTER_IMG_KV_PACK);
  }
  if (prof_detail && !do_img) {
    // Keep event timeline consistent so printing logic is simple.
    rec(EV_AFTER_IMG_K_PROJ);
    rec(EV_AFTER_IMG_V_PROJ);
    rec(EV_AFTER_IMG_KV_PACK);
  }

  // 4) Single FMHA over concatenated [text; image] KV
  cudaError_t fmha_err = run_fmha(Mk_total, reinterpret_cast<cutlass::half_t*>(out_acc));
  if (fmha_err != cudaSuccess) { return fmha_err; }
  rec(EV_AFTER_FMHA);

  // 5) Output projection (optionally fused with residual)
  cudaError_t out_err;
  if (fuse_residual) {
    out_err = cutlass_linear_layer_rrr_fused_residual(
        reinterpret_cast<const half*>(out_acc),
        reinterpret_cast<const half*>(p.w_out),
        reinterpret_cast<const half*>(p.b_out),
        reinterpret_cast<half*>(p.out_after_linear),
        M, K, K, stream);
  } else {
    out_err = cutlass_linear_layer_rrr(
        reinterpret_cast<const half*>(out_acc),
        reinterpret_cast<const half*>(p.w_out),
        reinterpret_cast<const half*>(p.b_out),
        reinterpret_cast<half*>(p.out_after_linear),
        M, K, K, stream);
  }
  if (out_err != cudaSuccess) { return out_err; }
  rec(EV_AFTER_OUT_PROJ);

  if (prof_detail) {
    cudaEventSynchronize(ev[EV_AFTER_OUT_PROJ]);
    auto ms = [&](int a, int b) -> float {
      float out = 0.f;
      cudaEventElapsedTime(&out, ev[a], ev[b]);
      return out;
    };
    float t_q_proj   = ms(EV_START, EV_AFTER_Q_PROJ);
    float t_q_pack   = ms(EV_AFTER_Q_PROJ, EV_AFTER_Q_PACK);
    float t_kv_text  = ms(EV_AFTER_Q_PACK, EV_AFTER_TEXT_KV_PROJ);
    float t_pack_txt = ms(EV_AFTER_TEXT_KV_PROJ, EV_AFTER_TEXT_KV_PACK);
    float t_k_img    = ms(EV_AFTER_TEXT_KV_PACK, EV_AFTER_IMG_K_PROJ);
    float t_v_img    = ms(EV_AFTER_IMG_K_PROJ, EV_AFTER_IMG_V_PROJ);
    float t_pack_img = ms(EV_AFTER_IMG_V_PROJ, EV_AFTER_IMG_KV_PACK);
    float t_fmha     = ms(EV_AFTER_IMG_KV_PACK, EV_AFTER_FMHA);
    float t_out      = ms(EV_AFTER_FMHA, EV_AFTER_OUT_PROJ);
    float t_total    = ms(EV_START, EV_AFTER_OUT_PROJ);

    std::printf(
      "[wan_ca_i2v_cutlass][call=%lld] M=%d K=%d H=%d D=%d Mk_text=%d Mk_img=%d | "
      "q_proj=%.3f q_pack=%.3f kv_text=%.3f pack_text=%.3f "
      "k_img=%.3f v_img=%.3f pack_img=%.3f fmha=%.3f out=%.3f total=%.3f ms\n",
      prof_call_idx, M, K, H, D, Mk_text, Mk_img,
      t_q_proj, t_q_pack, t_kv_text, t_pack_txt,
      t_k_img, t_v_img, t_pack_img, t_fmha, t_out, t_total
    );

    for (int i = 0; i < EV_COUNT; ++i) cudaEventDestroy(ev[i]);
  }

  return cudaSuccess;
}

// ============================================================================
// cuDNN Backend Implementation
// ============================================================================

// cuDNN handle and graph cache management (thread-local for safety)
namespace {
  thread_local cudnnHandle_t g_cudnn_handle = nullptr;

  cudnnHandle_t get_cudnn_handle() {
    if (!g_cudnn_handle) {
      cudnnCreate(&g_cudnn_handle);
    }
    return g_cudnn_handle;
  }

  // Cache for cuDNN graphs (key: B_Mq_H_D_Mk)
  struct GraphCacheKey {
    int B, Mq, H, D, Mk;
    bool operator==(const GraphCacheKey& other) const {
      return B == other.B && Mq == other.Mq && H == other.H && D == other.D && Mk == other.Mk;
    }
  };

  struct GraphCacheKeyHash {
    std::size_t operator()(const GraphCacheKey& k) const {
      size_t h = std::hash<int>()(k.B);
      h ^= (std::hash<int>()(k.Mq) << 1);
      h ^= (std::hash<int>()(k.H) << 2);
      h ^= (std::hash<int>()(k.D) << 3);
      h ^= (std::hash<int>()(k.Mk) << 4);
      return h;
    }
  };

  thread_local std::unordered_map<GraphCacheKey, std::shared_ptr<fe::graph::Graph>, GraphCacheKeyHash> g_sdpa_graph_cache;

  // Cache for packed-QKV bf16 cuDNN graphs (key: B_Mq_Mk_H_D_causal)
  struct PackedFmhaCacheKey {
    int B, Mq, Mk, H, D;
    bool causal;
    bool operator==(const PackedFmhaCacheKey& o) const {
      return B == o.B && Mq == o.Mq && Mk == o.Mk && H == o.H && D == o.D && causal == o.causal;
    }
  };

  struct PackedFmhaCacheKeyHash {
    std::size_t operator()(const PackedFmhaCacheKey& k) const {
      size_t h = std::hash<int>()(k.B);
      h ^= (std::hash<int>()(k.Mq)    << 1);
      h ^= (std::hash<int>()(k.Mk)    << 2);
      h ^= (std::hash<int>()(k.H)     << 3);
      h ^= (std::hash<int>()(k.D)     << 4);
      h ^= (std::hash<bool>()(k.causal) << 5);
      return h;
    }
  };

  struct PackedFp8FmhaCacheKey {
    int B, Mq, Mk, H, D;
    bool causal;
    std::string layout;
    std::string heuristics;
    std::string plan;
    bool operator==(const PackedFp8FmhaCacheKey& o) const {
      return B == o.B && Mq == o.Mq && Mk == o.Mk && H == o.H && D == o.D &&
             causal == o.causal && layout == o.layout &&
             heuristics == o.heuristics && plan == o.plan;
    }
  };

  struct PackedFp8FmhaCacheKeyHash {
    std::size_t operator()(const PackedFp8FmhaCacheKey& k) const {
      size_t h = std::hash<int>()(k.B);
      h ^= (std::hash<int>()(k.Mq) << 1);
      h ^= (std::hash<int>()(k.Mk) << 2);
      h ^= (std::hash<int>()(k.H) << 3);
      h ^= (std::hash<int>()(k.D) << 4);
      h ^= (std::hash<bool>()(k.causal) << 5);
      h ^= (std::hash<std::string>()(k.layout) << 6);
      h ^= (std::hash<std::string>()(k.heuristics) << 7);
      h ^= (std::hash<std::string>()(k.plan) << 8);
      return h;
    }
  };

  thread_local std::unordered_map<PackedFmhaCacheKey,
                                   std::shared_ptr<fe::graph::Graph>,
                                   PackedFmhaCacheKeyHash>
      g_packed_fmha_graph_cache;
  thread_local std::unordered_map<PackedFp8FmhaCacheKey,
                                   std::shared_ptr<fe::graph::Graph>,
                                   PackedFp8FmhaCacheKeyHash>
      g_packed_fp8_fmha_graph_cache;

  // Growable device workspace for packed-QKV cuDNN SDPA
  thread_local void*  g_packed_fmha_ws      = nullptr;
  thread_local size_t g_packed_fmha_ws_size = 0;

  static cudaError_t ensure_packed_fmha_ws(size_t needed) {
    if (needed <= g_packed_fmha_ws_size) return cudaSuccess;
    if (g_packed_fmha_ws) cudaFree(g_packed_fmha_ws);
    g_packed_fmha_ws = nullptr;
    cudaError_t err = cudaMalloc(&g_packed_fmha_ws, needed);
    if (err != cudaSuccess) return err;
    g_packed_fmha_ws_size = needed;
    return cudaSuccess;
  }
}

// cuDNN FMHA helper: accepts pre-packed BMHK Q/K/V in bf16, skips QKV projection and RoPE.
//
// Inputs  Q     [B*Mq, H, D]  bfloat16, contiguous (BMHK — batch items concatenated along M)
//         K, V  [B*Mk, H, D]  bfloat16, contiguous
// Output  O     [B*Mq, H, D]  bfloat16, caller-allocated
// Params  causal → top-left causal mask;  scale 0 → 1/sqrt(D)
cudaError_t run_cudnn_fmha_packed_qkv(
    const cutlass::bfloat16_t* Q,
    const cutlass::bfloat16_t* K,
    const cutlass::bfloat16_t* V,
    cutlass::bfloat16_t* O,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream) {
  if (Mq <= 0 || Mk <= 0) return cudaErrorInvalidValue;

  auto handle = get_cudnn_handle();
  cudnnSetStream(handle, stream);

  PackedFmhaCacheKey cache_key{B, Mq, Mk, H, D, causal};
  auto it = g_packed_fmha_graph_cache.find(cache_key);

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {1, const_cast<cutlass::bfloat16_t*>(Q)},
      {2, const_cast<cutlass::bfloat16_t*>(K)},
      {3, const_cast<cutlass::bfloat16_t*>(V)},
      {4, O}};

  auto build_graph = [&](fe::HeurMode_t heur_mode) -> std::shared_ptr<fe::graph::Graph> {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    // Strides for [B*Mq, H, D] packed layout viewed as [B, H, Mq, D]:
    //   batch stride = Mq*H*D,  head stride = D,  seq stride = H*D,  dim stride = 1
    int64_t K_total = int64_t(H) * D;
    auto tQ = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q").set_uid(1)
                                .set_dim   ({B, H, Mq, D})
                                .set_stride({int64_t(Mq) * K_total, D, K_total, 1}));
    auto tK = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K").set_uid(2)
                                .set_dim   ({B, H, Mk, D})
                                .set_stride({int64_t(Mk) * K_total, D, K_total, 1}));
    auto tV = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V").set_uid(3)
                                .set_dim   ({B, H, Mk, D})
                                .set_stride({int64_t(Mk) * K_total, D, K_total, 1}));

    float attn_scale = (scale > 0.f) ? scale : (1.0f / sqrtf(float(D)));
    auto sdpa_opts = fe::graph::SDPA_attributes()
                         .set_name("packed_sdpa")
                         .set_attn_scale(attn_scale)
                         .set_causal_mask(causal);

    auto [tO, tStats] = graph->sdpa(tQ, tK, tV, sdpa_opts);
    tO->set_output(true)
        .set_dim   ({B, H, Mq, D})
        .set_stride({int64_t(Mq) * K_total, D, K_total, 1})
        .set_uid(4);

    if (!graph->build(handle, {heur_mode}).is_good()) {
      return nullptr;
    }
    return graph;
  };

  auto execute_graph = [&](const std::shared_ptr<fe::graph::Graph>& graph) -> cudaError_t {
    int64_t workspace_size = 0;
    if (!graph->get_workspace_size(workspace_size).is_good()) {
      return cudaErrorUnknown;
    }
    if (cudaError_t ws_err = ensure_packed_fmha_ws((size_t)workspace_size); ws_err != cudaSuccess) {
      return ws_err;
    }

    if (!graph->execute(handle, variant_pack, g_packed_fmha_ws).is_good()) {
      return cudaErrorUnknown;
    }
    return cudaSuccess;
  };

  if (it != g_packed_fmha_graph_cache.end()) {
    cudaError_t cached_err = execute_graph(it->second);
    if (cached_err == cudaSuccess) {
      return cudaSuccess;
    }
    g_packed_fmha_graph_cache.erase(it);
    if (cached_err != cudaErrorUnknown) {
      return cached_err;
    }
  }

  const fe::HeurMode_t heur_modes[] = {
      fe::HeurMode_t::A,
      fe::HeurMode_t::B,
      fe::HeurMode_t::FALLBACK};
  for (fe::HeurMode_t heur_mode : heur_modes) {
    auto graph = build_graph(heur_mode);
    if (!graph) {
      continue;
    }
    cudaError_t err = execute_graph(graph);
    if (err == cudaSuccess) {
      g_packed_fmha_graph_cache[cache_key] = graph;
      return cudaSuccess;
    }
    if (err != cudaErrorUnknown) {
      return err;
    }
  }
  return cudaErrorUnknown;
}

cudaError_t run_cudnn_fmha_packed_qkv_fp8(
    const cutlass::float_e4m3_t* Q,
    const cutlass::float_e4m3_t* K,
    const cutlass::float_e4m3_t* V,
    cutlass::float_e4m3_t* O,
    const float* descale_q,
    const float* descale_k,
    const float* descale_v,
    const float* descale_s,
    const float* scale_s,
    const float* scale_o,
    float* amax_s,
    float* amax_o,
    int B, int Mq, int Mk,
    int H, int D,
    bool causal,
    float scale,
    cudaStream_t stream) {
  if (!Q || !K || !V || !O || !descale_q || !descale_k || !descale_v ||
      !descale_s || !scale_s || !scale_o || !amax_s || !amax_o ||
      B <= 0 || Mq <= 0 || Mk <= 0 || H <= 0 || D <= 0) {
    return cudaErrorInvalidValue;
  }

  auto handle = get_cudnn_handle();
  cudnnSetStream(handle, stream);

  const auto selection = select_cosmos_fp8_sdpa(B, Mq, Mk, H, D);
  PackedFp8FmhaCacheKey cache_key{
      B, Mq, Mk, H, D, causal,
      selection.layout, selection.heuristics, selection.plan};
  auto it = g_packed_fp8_fmha_graph_cache.find(cache_key);
  std::shared_ptr<fe::graph::Graph> graph;
  bool needs_autotune = false;
  bool cache_miss = false;

  if (it != g_packed_fp8_fmha_graph_cache.end()) {
    graph = it->second;
  } else {
    cache_miss = true;
    graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::FP8_E4M3)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t hidden_stride = int64_t(H) * D;
    const bool input_bhmd = selection.layout == "bhmd";
    const int64_t q_batch_stride = input_bhmd ? int64_t(H) * Mq * D : int64_t(Mq) * hidden_stride;
    const int64_t k_batch_stride = input_bhmd ? int64_t(H) * Mk * D : int64_t(Mk) * hidden_stride;
    const int64_t q_head_stride = input_bhmd ? int64_t(Mq) * D : D;
    const int64_t k_head_stride = input_bhmd ? int64_t(Mk) * D : D;
    const int64_t q_seq_stride = input_bhmd ? D : hidden_stride;
    const int64_t k_seq_stride = input_bhmd ? D : hidden_stride;
    auto tQ = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q").set_uid(1)
                                .set_dim({B, H, Mq, D})
                                .set_stride({q_batch_stride, q_head_stride, q_seq_stride, 1})
                                .set_data_type(fe::DataType_t::FP8_E4M3));
    auto tK = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K").set_uid(2)
                                .set_dim({B, H, Mk, D})
                                .set_stride({k_batch_stride, k_head_stride, k_seq_stride, 1})
                                .set_data_type(fe::DataType_t::FP8_E4M3));
    auto tV = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V").set_uid(3)
                                .set_dim({B, H, Mk, D})
                                .set_stride({k_batch_stride, k_head_stride, k_seq_stride, 1})
                                .set_data_type(fe::DataType_t::FP8_E4M3));

    auto descale_q_t = graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("Descale_Q").set_uid(5)
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::FLOAT));
    auto descale_k_t = graph->tensor_like(descale_q_t, "Descale_K");
    descale_k_t->set_uid(6);
    auto descale_v_t = graph->tensor_like(descale_q_t, "Descale_V");
    descale_v_t->set_uid(7);
    auto descale_s_t = graph->tensor_like(descale_q_t, "Descale_S");
    descale_s_t->set_uid(8);
    auto scale_s_t = graph->tensor_like(descale_q_t, "Scale_S");
    scale_s_t->set_uid(9);
    auto scale_o_t = graph->tensor_like(descale_q_t, "Scale_O");
    scale_o_t->set_uid(10);

    float attn_scale = (scale > 0.f) ? scale : (1.0f / sqrtf(float(D)));
    auto sdpa_fp8_options = fe::graph::SDPA_fp8_attributes()
                                .set_name("packed_sdpa_fp8_probe")
                                .set_generate_stats(false)
                                .set_causal_mask(causal)
                                .set_attn_scale(attn_scale);

    auto [tO, tStats, tAmaxS, tAmaxO] = graph->sdpa_fp8(
        tQ, tK, tV,
        descale_q_t, descale_k_t, descale_v_t, descale_s_t,
        scale_s_t, scale_o_t,
        sdpa_fp8_options);
    if (tStats != nullptr) {
      return cudaErrorNotSupported;
    }

    tO->set_output(true)
        .set_dim({B, H, Mq, D})
        .set_stride({int64_t(Mq) * hidden_stride, D, hidden_stride, 1})
        .set_data_type(fe::DataType_t::FP8_E4M3)
        .set_uid(4);
    tAmaxS->set_output(true)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT)
        .set_uid(11);
    tAmaxO->set_output(true)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT)
        .set_uid(12);

    if (!graph->build(
             handle,
             cosmos_fp8_sdpa_heur_modes(selection.heuristics),
             cosmos_fp8_sdpa_build_policy(selection.plan)).is_good()) {
      return cudaErrorNotSupported;
    }
    needs_autotune = selection.plan == "autotune";
  }

  int64_t workspace_size = 0;
  if (!graph->get_workspace_size(workspace_size).is_good()) {
    return cudaErrorUnknown;
  }
  if (needs_autotune) {
    workspace_size = std::max<int64_t>(workspace_size, graph->get_autotune_workspace_size());
  }
  if (cudaError_t ws_err = ensure_packed_fmha_ws((size_t)workspace_size); ws_err != cudaSuccess) {
    return ws_err;
  }

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {1, const_cast<cutlass::float_e4m3_t*>(Q)},
      {2, const_cast<cutlass::float_e4m3_t*>(K)},
      {3, const_cast<cutlass::float_e4m3_t*>(V)},
      {4, O},
      {5, const_cast<float*>(descale_q)},
      {6, const_cast<float*>(descale_k)},
      {7, const_cast<float*>(descale_v)},
      {8, const_cast<float*>(descale_s)},
      {9, const_cast<float*>(scale_s)},
      {10, const_cast<float*>(scale_o)},
      {11, amax_s},
      {12, amax_o}};

  if (needs_autotune) {
    if (!graph->autotune(handle, variant_pack, g_packed_fmha_ws).is_good()) {
      return cudaErrorUnknown;
    }
  }
  if (cache_miss) {
    g_packed_fp8_fmha_graph_cache[cache_key] = graph;
  }

  if (!graph->execute(handle, variant_pack, g_packed_fmha_ws).is_good()) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

// cuDNN self-attention implementation using cudnn-frontend SDPA
template <typename WeightT>
static cudaError_t run_self_attention_cudnn_impl(
    const AttentionDeviceParamsT<WeightT>& p,
    cudaStream_t stream,
    cutlass::half_t* qkv_capture = nullptr,
    float* act_scale_capture = nullptr) {
  int B = p.B;
  int Mq = p.Mq;
  if (Mq <= 0) { return cudaErrorInvalidValue; }
  int M = B * Mq;
  int K = p.K, H = p.H, D = p.D;

  // Optional fine-grained profiling:
  // - GPU time via CUDA events for qkv/pack/sdpa/out
  // - CPU wall time for cuDNN graph build (first-call overhead)
  long long prof_call_idx = 0;
  bool prof_detail = wan_profile_detail_enabled(&prof_call_idx);
  enum {
    EV_START = 0,
    EV_AFTER_QKV_PROJ,
    EV_AFTER_PACK_ROPE,
    EV_BEFORE_SDPA_EXEC,
    EV_AFTER_SDPA_EXEC,
    EV_AFTER_OUT,
    EV_COUNT
  };
  cudaEvent_t ev[EV_COUNT];
  auto rec = [&](int idx) {
    if (prof_detail) cudaEventRecord(ev[idx], stream);
  };
  if (prof_detail) {
    for (int i = 0; i < EV_COUNT; ++i) cudaEventCreate(&ev[i]);
    rec(EV_START);
  }
  double graph_build_ms_cpu = 0.0;
  bool graph_built_now = false;

  // Use cuDNN workspace buffers
  cutlass::half_t* qkv_row = p.workspace->cudnn.sa_qkv_row;
  cutlass::half_t* dQh_bmhk = p.workspace->cudnn.sa_q_bmhk;
  cutlass::half_t* dKh_bmhk = p.workspace->cudnn.sa_k_bmhk;
  cutlass::half_t* dVh_bmhk = p.workspace->cudnn.sa_v_bmhk;
  cutlass::half_t* dOh_bmhk = p.workspace->cudnn.sa_o_bmhk;

  // 1) QKV projection via CUTLASS linear layer
  {
    // Use dQh_bmhk as fp8_scratch (will be overwritten in the next step)
    cudaError_t err = apply_linear_row<WeightT>(
        p.hidden_states,
        p.w_qkv,
        p.b_qkv,
        qkv_row,
        M, K, 3 * K, stream,
        dQh_bmhk,  // fp8_scratch
        p.w_qkv_scale,  // weight_scale (FP8)
        p.workspace->int8_scratch,  // int8_scratch
        p.workspace->int8_act_block_scales,  // int8_act_block_scales (INT8)
        p.w_qkv_block_scale);  // weight_block_scale (INT8)
    if (err != cudaSuccess) { return err; }
  }
  if (qkv_capture) {
    auto err = cudaMemcpyAsync(qkv_capture, qkv_row, sizeof(cutlass::half_t) * size_t(M) * 3 * K, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) { return err; }
  }
  if (act_scale_capture && std::is_same<WeightT, int8_t>::value) {
    float act_scale = 1.0f;
    cudaError_t act = compute_activation_scale(
      reinterpret_cast<const cutlass::half_t*>(p.hidden_states),
      int64_t(M) * K,
      &act_scale,
      stream);
    if (act != cudaSuccess) return act;
    auto err = cudaMemcpyAsync(act_scale_capture, &act_scale, sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
  }
  rec(EV_AFTER_QKV_PROJ);

  // 2) Pack and normalize Q, K, V + apply RoPE
  {
    rmsnorm_pack_bmhk_rope_from_row_kernel<<<M, 256, 2*256*sizeof(float), stream>>>(
      reinterpret_cast<const half*>(qkv_row), M, K, H, D, Mq,
      p.norm_q_gamma, p.norm_k_gamma, 1.0e-6f,
      p.rotary_cos, p.rotary_sin,
      dQh_bmhk, dKh_bmhk, dVh_bmhk);
    CUDA_CHECK(cudaGetLastError());
  }
  rec(EV_AFTER_PACK_ROPE);

  // 3) Run cuDNN SDPA
  {
    auto handle = get_cudnn_handle();
    cudnnSetStream(handle, stream);

    // Check graph cache or create new graph
    GraphCacheKey cache_key{B, Mq, H, D, Mq};  // Mk = Mq for self-attention
    auto it = g_sdpa_graph_cache.find(cache_key);
    std::shared_ptr<fe::graph::Graph> graph;

    if (it != g_sdpa_graph_cache.end()) {
      graph = it->second;
    } else {
      auto t0 = std::chrono::high_resolution_clock::now();
      graph_built_now = true;
      // Create cuDNN graph for SDPA
      graph = std::make_shared<fe::graph::Graph>();
      graph->set_io_data_type(fe::DataType_t::HALF)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // Define Q, K, V tensors in BHSD format (batch=B, heads=H, seq=Mq, dim=D)
    // Note: The packing kernel outputs row-major [M, K] where K=H*D
    // So strides are [batch_stride (unused), D, H*D, 1] to match row-major layout
    int64_t b = B;
    int64_t K_total = H * D;  // Total hidden dimension
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(1)
                               .set_dim({b, H, Mq, D})
                               .set_stride({int64_t(Mq) * K_total, D, K_total, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(2)
                               .set_dim({b, H, Mq, D})
                               .set_stride({int64_t(Mq) * K_total, D, K_total, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(3)
                               .set_dim({b, H, Mq, D})
                               .set_stride({int64_t(Mq) * K_total, D, K_total, 1}));

    // SDPA options with attention scale
    float attn_scale = 1.0f / sqrtf(static_cast<float>(D));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("sdpa_self_attention")
                            .set_attn_scale(attn_scale);

    // Create SDPA operation
    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

      // Set output tensor properties with row-major [M, K] layout to match input format
      O->set_output(true).set_dim({b, H, Mq, D}).set_stride({int64_t(Mq) * K_total, D, K_total, 1}).set_uid(4);

      // Build the graph
      if (!graph->build(handle, {fe::HeurMode_t::A}).is_good()) {
        return cudaErrorUnknown;
      }

      // Cache the graph
      g_sdpa_graph_cache[cache_key] = graph;
      auto t1 = std::chrono::high_resolution_clock::now();
      graph_build_ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Prepare variant pack
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {1, dQh_bmhk}, {2, dKh_bmhk}, {3, dVh_bmhk}, {4, dOh_bmhk}};

    // Get workspace size and validate
    int64_t workspace_size = 0;
    if (!graph->get_workspace_size(workspace_size).is_good()) {
      return cudaErrorUnknown;
    }

    // Ensure cuDNN SDPA workspace exists with sufficient capacity
    p.workspace->ensure_cudnn_workspace((size_t)workspace_size);
    void* workspace = p.workspace->cudnn.cudnn_workspace;
    rec(EV_BEFORE_SDPA_EXEC);
    if (!graph->execute(handle, variant_pack, workspace).is_good()) {
      return cudaErrorUnknown;
    }
    rec(EV_AFTER_SDPA_EXEC);
  }

  // 4) Output projection - dOh_bmhk is already in row-major [M, K] format due to stride config
  {
    // INT8 fused gated residual path: GEMM + gate*output + residual in one kernel
    if constexpr (std::is_same_v<WeightT, int8_t>) {
      if (p.gate_sst != nullptr) {
        // Copy residual source into output buffer before GEMM overwrites it
        cudaMemcpyAsync(p.out_after_linear, p.residual_src,
            size_t(M) * K * sizeof(cutlass::half_t),
            cudaMemcpyDeviceToDevice, stream);
        // Quantize activations
        cudaError_t q = quantize_per_block_128(
            reinterpret_cast<const half*>(dOh_bmhk),
            p.workspace->int8_scratch,
            p.workspace->int8_act_block_scales,
            M, K, stream, false);
        if (q != cudaSuccess) return q;
        // Fused GEMM + gated residual
        cudaError_t err = int8_gemm_gated_residual(
            p.workspace->int8_scratch,
            p.workspace->int8_act_block_scales,
            reinterpret_cast<const int8_t*>(p.w_out),
            p.w_out_block_scale,
            reinterpret_cast<half*>(p.out_after_linear),
            reinterpret_cast<const half*>(p.b_out),
            reinterpret_cast<const half*>(p.gate_sst),
            reinterpret_cast<const half*>(p.gate_temb),
            p.gate_temb_row_stride,
            p.gate_idx,
            M, K, K, stream);
        if (err != cudaSuccess) return err;
        return cudaSuccess;
      }
    }
    // Fallback: existing apply_linear_row path (FP16/FP8, or INT8 without gate fusion)
    cudaError_t err = apply_linear_row<WeightT>(
        dOh_bmhk,
        p.w_out,
        p.b_out,
        p.out_after_linear,
        M, K, K, stream,
        dQh_bmhk,  // fp8_scratch
        p.w_out_scale,  // weight_scale (FP8)
        p.workspace->int8_scratch,  // int8_scratch
        p.workspace->int8_act_block_scales,  // int8_act_block_scales (INT8)
        p.w_out_block_scale);  // weight_block_scale (INT8)
    if (err != cudaSuccess) { return err; }
  }
  rec(EV_AFTER_OUT);

  if (prof_detail) {
    cudaEventSynchronize(ev[EV_AFTER_OUT]);
    auto ms = [&](int a, int b) -> float {
      float out = 0.f;
      cudaEventElapsedTime(&out, ev[a], ev[b]);
      return out;
    };
    float t_qkv  = ms(EV_START, EV_AFTER_QKV_PROJ);
    float t_pack = ms(EV_AFTER_QKV_PROJ, EV_AFTER_PACK_ROPE);
    float t_sdpa = ms(EV_BEFORE_SDPA_EXEC, EV_AFTER_SDPA_EXEC);
    float t_out  = ms(EV_AFTER_SDPA_EXEC, EV_AFTER_OUT);
    float t_tot  = ms(EV_START, EV_AFTER_OUT);
    std::printf(
      "[wan_sa_cudnn][call=%lld] M=%d K=%d H=%d D=%d | "
      "qkv=%.3f pack_rope=%.3f sdpa=%.3f out=%.3f total=%.3f ms",
      prof_call_idx, M, K, H, D, t_qkv, t_pack, t_sdpa, t_out, t_tot
    );
    if (graph_built_now) {
      std::printf(" | graph_build_cpu=%.3f ms", graph_build_ms_cpu);
    }
    std::printf("\n");
    for (int i = 0; i < EV_COUNT; ++i) cudaEventDestroy(ev[i]);
  }

  return cudaSuccess;
}
// cuDNN cross-attention implementation using cudnn-frontend SDPA
template <typename WeightT>
static cudaError_t run_cross_attention_cudnn(const AttentionDeviceParamsT<WeightT>& p, bool fuse_residual, cudaStream_t stream) {
  int B = p.B;
  int Mq = p.Mq;
  if (Mq <= 0) { return cudaErrorInvalidValue; }
  int M = B * Mq;
  int K = p.K, H = p.H, D = p.D;
  int Mk = p.Mk ? p.Mk : Mq;

  // Determine encoder batch size for KV projection (shared KV optimization for CFG)
  int enc_B = (p.encoder_batch_size > 0) ? p.encoder_batch_size : B;
  bool shared_kv = (enc_B < B);

  // Use cuDNN workspace buffers
  cutlass::half_t* dQh_bmhk = p.workspace->cudnn.ca_q_bmhk;
  cutlass::half_t* dKh_bmhk = p.workspace->cudnn.ca_k_bmhk;
  cutlass::half_t* dVh_bmhk = p.workspace->cudnn.ca_v_bmhk;
  cutlass::half_t* dOh_bmhk = p.workspace->cudnn.ca_o_bmhk;
  cutlass::half_t* dQrow = p.workspace->cudnn.ca_q_row;
  cutlass::half_t* dKVrow = p.workspace->cudnn.ca_kv_row;

  CA_NAN_CHECK(p.hidden_states, int64_t(M)*K, "CA_input_hidden_states");
  CA_NAN_CHECK(p.encoder_hidden_states, int64_t(enc_B)*Mk*K, "CA_input_encoder_hidden_states");

  // 1a) Q projection
  {
    // Use dQh_bmhk as fp8_scratch (will be overwritten in packing step)
    cudaError_t err = apply_linear_row<WeightT>(
        p.hidden_states,
        p.w_q,
        p.b_q,
        dQrow,
        M, K, K, stream,
        dQh_bmhk,  // fp8_scratch
        p.w_q_scale,  // weight_scale (FP8)
        p.workspace->int8_scratch,  // int8_scratch
        p.workspace->int8_act_block_scales,  // int8_act_block_scales (INT8)
        p.w_q_block_scale);  // weight_block_scale (INT8)
    if (err != cudaSuccess) { return err; }
    CA_NAN_CHECK(dQrow, int64_t(M)*K, "CA_after_Q_gemm");
    rmsnorm_pack_single_from_row_kernel<<<M, 256, 256*sizeof(float), stream>>>(
      reinterpret_cast<const half*>(dQrow), M, K, H, D,
      p.norm_q_gamma, 1.0e-6f,
      dQh_bmhk);
    CUDA_CHECK(cudaGetLastError());
    CA_NAN_CHECK(dQh_bmhk, int64_t(M)*K, "CA_after_Q_rmsnorm_pack");
  }

  // 1b) KV projection - project only enc_B*Mk tokens when shared_kv=true
  {
    // Use dKh_bmhk as fp8_scratch (will be overwritten in packing step)
    int kv_rows = enc_B * Mk;
    cudaError_t err = apply_linear_row<WeightT>(
        p.encoder_hidden_states,
        p.w_kv,
        p.b_kv,
        dKVrow,
        kv_rows, K, 2 * K, stream,
        dKh_bmhk,  // fp8_scratch
        p.w_kv_scale,  // weight_scale (FP8)
        p.workspace->int8_scratch,  // int8_scratch
        p.workspace->int8_act_block_scales,  // int8_act_block_scales (INT8)
        p.w_kv_block_scale);  // weight_block_scale (INT8)
    if (err != cudaSuccess) { return err; }
    CA_NAN_CHECK(dKVrow, int64_t(kv_rows)*2*K, "CA_after_KV_gemm");

    rmsnorm_pack_kv_from_row_kernel<<<kv_rows, 256, 256*sizeof(float), stream>>>(
      reinterpret_cast<const half*>(dKVrow), kv_rows, K, H, D,
      p.norm_k_gamma, 1.0e-6f,
      dKh_bmhk, dVh_bmhk);
    CUDA_CHECK(cudaGetLastError());
    CA_NAN_CHECK(dKh_bmhk, int64_t(kv_rows)*K, "CA_after_K_rmsnorm_pack");
    CA_NAN_CHECK(dVh_bmhk, int64_t(kv_rows)*K, "CA_after_V_pack");
  }

  // 2) Run cuDNN cross-attention SDPA
  {
    auto handle = get_cudnn_handle();
    cudnnSetStream(handle, stream);

    // Check graph cache or create new graph
    // Include shared_kv in cache key since stride patterns differ
    GraphCacheKey cache_key{B, Mq, H, D, shared_kv ? -Mk : Mk};  // Negative Mk signals shared KV mode
    auto it = g_sdpa_graph_cache.find(cache_key);
    std::shared_ptr<fe::graph::Graph> graph;

    if (it != g_sdpa_graph_cache.end()) {
      graph = it->second;
    } else {
      // Create cuDNN graph for cross-attention SDPA
      graph = std::make_shared<fe::graph::Graph>();
      graph->set_io_data_type(fe::DataType_t::HALF)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // Define Q, K, V tensors in BHSD format
    // Note: Packing kernels output row-major [M, K] or [Mk, K] where K=H*D
    // Q: batch=B, heads=H, seq=Mq, dim=D
    // K, V: batch=B (logical), heads=H, seq=Mk (encoder sequence), dim=D
    //       When shared_kv=true, K/V batch stride=0 to broadcast single KV set
    int64_t b = B;
    int64_t K_total = H * D;  // Total hidden dimension
    int64_t kv_batch_stride = shared_kv ? 0 : (int64_t(Mk) * K_total);
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(1)
                               .set_dim({b, H, Mq, D})
                               .set_stride({int64_t(Mq) * K_total, D, K_total, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(2)
                               .set_dim({b, H, Mk, D})
                               .set_stride({kv_batch_stride, D, K_total, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(3)
                               .set_dim({b, H, Mk, D})
                               .set_stride({kv_batch_stride, D, K_total, 1}));

    // SDPA options with attention scale
    float attn_scale = 1.0f / sqrtf(static_cast<float>(D));
    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("sdpa_cross_attention")
                            .set_attn_scale(attn_scale);

    // Create SDPA operation
    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

      // Set output tensor properties with row-major [M, K] layout to match Q input format
      O->set_output(true).set_dim({b, H, Mq, D}).set_stride({int64_t(Mq) * K_total, D, K_total, 1}).set_uid(4);

      // Build the graph
      if (!graph->build(handle, {fe::HeurMode_t::A}).is_good()) {
        return cudaErrorUnknown;
      }

      // Cache the graph
      g_sdpa_graph_cache[cache_key] = graph;
    }

    // Prepare variant pack
    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {1, dQh_bmhk}, {2, dKh_bmhk}, {3, dVh_bmhk}, {4, dOh_bmhk}};

    // Get workspace size and validate
    int64_t workspace_size = 0;
    if (!graph->get_workspace_size(workspace_size).is_good()) {
      return cudaErrorUnknown;
    }

    // Ensure cuDNN SDPA workspace exists with sufficient capacity
    p.workspace->ensure_cudnn_workspace((size_t)workspace_size);
    void* workspace = p.workspace->cudnn.cudnn_workspace;
    if (!graph->execute(handle, variant_pack, workspace).is_good()) {
      return cudaErrorUnknown;
    }
  }
  CA_NAN_CHECK(dOh_bmhk, int64_t(M)*K, "CA_after_cuDNN_SDPA");

  // 3) Output projection - dOh_bmhk is already in row-major [M, K] format due to stride config
  {
    // Use dQh_bmhk as fp8_scratch.
    // For temp_out (FP8 scaled path): need M*K halves, but dKVrow is only Mk*2K which is
    // too small when M >> Mk. Use sa_qkv_row (M*3K halves, self-attention is done).
    cutlass::half_t* out_temp = p.workspace->cudnn.sa_qkv_row;  // M*3K >= M*K
    cudaError_t err = cudaSuccess;
    if (fuse_residual) {
      err = apply_linear_row_fused_residual<WeightT>(
          dOh_bmhk,
          p.w_out,
          p.b_out,
          p.out_after_linear,  // residual_inout
          M, K, K, stream,
          dQh_bmhk,  // fp8_scratch
          p.w_out_scale,  // weight_scale (FP8)
          out_temp,  // temp_out (M*K GEMM output, needs M*K halves)
          p.workspace->int8_scratch,  // int8_scratch
          p.workspace->int8_act_block_scales,  // int8_act_block_scales (INT8)
          p.w_out_block_scale);  // weight_block_scale (INT8)
    } else {
      err = apply_linear_row<WeightT>(
          dOh_bmhk,
          p.w_out,
          p.b_out,
          p.out_after_linear,
          M, K, K, stream,
          dQh_bmhk,  // fp8_scratch
          p.w_out_scale,  // weight_scale (FP8)
          p.workspace->int8_scratch,  // int8_scratch
          p.workspace->int8_act_block_scales,  // int8_act_block_scales (INT8)
          p.w_out_block_scale);  // weight_block_scale (INT8)
    }
    if (err != cudaSuccess) { return err; }
  }
  CA_NAN_CHECK(p.out_after_linear, int64_t(M)*K, "CA_after_out_proj_fused_residual");

  return cudaSuccess;
}


// cuDNN cross-attention implementation for fused text+image (I2V) path
template <typename WeightT>
static cudaError_t run_cross_attention_i2v_cudnn(const CrossAttentionI2VParamsT<WeightT>& p, bool fuse_residual, cudaStream_t stream) {
  int B = p.B;
  int Mq = p.M;                // query length per batch item
  int M_total = B * Mq;        // total query rows in flattened [B*Mq, K]
  int K = p.K, H = p.H, D = p.D;
  int Mk_text = p.Mk_text ? p.Mk_text : Mq;  // text KV length per batch item

  // Image branch enabled only when all required pointers are present.
  bool do_img = (p.Mk_img > 0) &&
                (p.encoder_hidden_states_img != nullptr) &&
                (p.w_add_k != nullptr) && (p.w_add_v != nullptr) &&
                (p.norm_added_k_gamma != nullptr);
  int Mk_img = do_img ? p.Mk_img : 0;
  int Mk_total = Mk_text + Mk_img;  // KV length per batch item (text + image)
  int Mk_text_total = B * Mk_text;
  int Mk_img_total = B * Mk_img;

  // Optional fine-grained profiling:
  // - GPU time via CUDA events around key kernels and SDPA execute
  // - CPU wall time for cuDNN graph build (first-call overhead)
  long long prof_call_idx = 0;
  bool prof_detail = wan_profile_detail_enabled(&prof_call_idx);
  enum {
    EV_START = 0,
    EV_AFTER_Q_PROJ,
    EV_AFTER_Q_PACK,
    EV_AFTER_TEXT_KV_PROJ,
    EV_AFTER_TEXT_KV_PACK,
    EV_AFTER_IMG_K_PROJ,
    EV_AFTER_IMG_V_PROJ,
    EV_AFTER_IMG_KV_PACK,
    EV_BEFORE_SDPA_EXEC,
    EV_AFTER_SDPA_EXEC,
    EV_AFTER_OUT_PROJ,
    EV_COUNT
  };
  cudaEvent_t ev[EV_COUNT];
  auto rec = [&](int idx) {
    if (prof_detail) cudaEventRecord(ev[idx], stream);
  };
  if (prof_detail) {
    for (int i = 0; i < EV_COUNT; ++i) cudaEventCreate(&ev[i]);
    rec(EV_START);
  }
  double graph_build_ms_cpu = 0.0;
  bool graph_built_now = false;

  // Workspace pointers (cuDNN layout)
  cutlass::half_t* dQh_bmhk = p.workspace->cudnn.ca_q_bmhk;   // [B*Mq, K]
  cutlass::half_t* dKh_bmhk = p.workspace->cudnn.ca_k_bmhk;   // [Mk_max, K] (Mk_max >= B*(Mk_text+Mk_img))
  cutlass::half_t* dVh_bmhk = p.workspace->cudnn.ca_v_bmhk;   // [Mk_max, K]
  cutlass::half_t* dOh_bmhk = p.workspace->cudnn.ca_o_bmhk;   // [B*Mq, K]
  cutlass::half_t* dQrow    = p.workspace->cudnn.ca_q_row;    // [B*Mq, K]
  cutlass::half_t* dKVrow   = p.workspace->cudnn.ca_kv_row;   // [Mk_max, 2K]

  // 1) Q projection + RMSNorm/pack
  {
    auto* fp8_scratch = reinterpret_cast<cutlass::half_t*>(dQh_bmhk); // safe scratch
    cudaError_t err = apply_linear_row<WeightT>(
        reinterpret_cast<const cutlass::half_t*>(p.hidden_states),
        reinterpret_cast<const WeightT*>(p.w_q),
        reinterpret_cast<const cutlass::half_t*>(p.b_q),
        reinterpret_cast<cutlass::half_t*>(dQrow),
        M_total, K, K, stream,
        fp8_scratch,
        reinterpret_cast<const cutlass::half_t*>(p.w_q_scale));
    if (err != cudaSuccess) { return err; }
    rec(EV_AFTER_Q_PROJ);
    rmsnorm_pack_single_from_row_kernel<<<M_total, 256, 256*sizeof(float), stream>>>(
      reinterpret_cast<const half*>(dQrow), M_total, K, H, D,
      p.norm_q_gamma, 1.0e-6f,
      dQh_bmhk);
    CUDA_CHECK(cudaGetLastError());
    rec(EV_AFTER_Q_PACK);
  }

  // 2) Text KV projection -> pack (for all batches)
  {
    auto* fp8_scratch = reinterpret_cast<cutlass::half_t*>(dKh_bmhk); // safe scratch
    cudaError_t err = apply_linear_row<WeightT>(
        reinterpret_cast<const cutlass::half_t*>(p.encoder_hidden_states_text),
        reinterpret_cast<const WeightT*>(p.w_kv),
        reinterpret_cast<const cutlass::half_t*>(p.b_kv),
        reinterpret_cast<cutlass::half_t*>(dKVrow),
        Mk_text_total, K, 2 * K, stream,
        fp8_scratch,
        reinterpret_cast<const cutlass::half_t*>(p.w_kv_scale));
    if (err != cudaSuccess) { return err; }
    rec(EV_AFTER_TEXT_KV_PROJ);

    // Pack into [B, Mk_total, K] layout in the shared buffer:
    // For each batch b: write text tokens into rows [b*Mk_total + 0 : b*Mk_total + Mk_text)
    for (int b = 0; b < B; ++b) {
      const half* kv_row_b = reinterpret_cast<const half*>(dKVrow) + size_t(b) * Mk_text * 2 * K;
      cutlass::half_t* dKh_b = dKh_bmhk + size_t(b) * Mk_total * K;
      cutlass::half_t* dVh_b = dVh_bmhk + size_t(b) * Mk_total * K;
      rmsnorm_pack_kv_from_row_kernel<<<Mk_text, 256, 256*sizeof(float), stream>>>(
        kv_row_b, Mk_text, K, H, D,
        p.norm_k_gamma, 1.0e-6f,
        dKh_b, dVh_b);
      CUDA_CHECK(cudaGetLastError());
    }
    rec(EV_AFTER_TEXT_KV_PACK);
  }

  // 3) Image KV projection -> pack into the tail after text (optional)
  if (do_img) {
    CUDA_CHECK(p.added_kv_proj_dim > 0 ? cudaSuccess : cudaErrorInvalidValue);

    // Store image KV rows in a contiguous region after the text KV rows:
    // kv_row_img_all is [B*Mk_img, 2K] row-major, arranged batch-major.
    half* kv_row_img_all = reinterpret_cast<half*>(dKVrow + size_t(Mk_text_total) * 2 * K);
    // Use a contiguous temp buffer for scaled K/V, then scatter into strided [B*Mk_img, 2K].
    auto* fp8_scratch = reinterpret_cast<cutlass::half_t*>(p.workspace->scratch_mk_b);
    auto* tmp_row = reinterpret_cast<cutlass::half_t*>(dQrow);
    cudaError_t errk = apply_linear_row<WeightT>(
        reinterpret_cast<const cutlass::half_t*>(p.encoder_hidden_states_img),
        reinterpret_cast<const WeightT*>(p.w_add_k),
        reinterpret_cast<const cutlass::half_t*>(p.b_add_k),
        reinterpret_cast<cutlass::half_t*>(tmp_row),
        Mk_img_total, p.added_kv_proj_dim, K, stream,
        fp8_scratch,
        reinterpret_cast<const cutlass::half_t*>(p.w_add_k_scale));
    if (errk != cudaSuccess) { return errk; }
    rec(EV_AFTER_IMG_K_PROJ);
    cudaError_t cpy_k = cudaMemcpy2DAsync(
        kv_row_img_all, size_t(2 * K) * sizeof(half),
        tmp_row, size_t(K) * sizeof(half),
        size_t(K) * sizeof(half), Mk_img_total,
        cudaMemcpyDeviceToDevice, stream);
    if (cpy_k != cudaSuccess) { return cpy_k; }

    cudaError_t errv = apply_linear_row<WeightT>(
        reinterpret_cast<const cutlass::half_t*>(p.encoder_hidden_states_img),
        reinterpret_cast<const WeightT*>(p.w_add_v),
        reinterpret_cast<const cutlass::half_t*>(p.b_add_v),
        reinterpret_cast<cutlass::half_t*>(tmp_row),
        Mk_img_total, p.added_kv_proj_dim, K, stream,
        fp8_scratch,
        reinterpret_cast<const cutlass::half_t*>(p.w_add_v_scale));
    if (errv != cudaSuccess) { return errv; }
    rec(EV_AFTER_IMG_V_PROJ);
    cudaError_t cpy_v = cudaMemcpy2DAsync(
        kv_row_img_all + K, size_t(2 * K) * sizeof(half),
        tmp_row, size_t(K) * sizeof(half),
        size_t(K) * sizeof(half), Mk_img_total,
        cudaMemcpyDeviceToDevice, stream);
    if (cpy_v != cudaSuccess) { return cpy_v; }

    // Pack into [B, Mk_total, K] layout:
    // For each batch b: write image tokens into rows [b*Mk_total + Mk_text : b*Mk_total + Mk_total)
    for (int b = 0; b < B; ++b) {
      const half* kv_row_img_b = kv_row_img_all + size_t(b) * Mk_img * 2 * K;
      cutlass::half_t* dKh_img_b = dKh_bmhk + (size_t(b) * Mk_total + Mk_text) * K;
      cutlass::half_t* dVh_img_b = dVh_bmhk + (size_t(b) * Mk_total + Mk_text) * K;
      rmsnorm_pack_kv_from_row_kernel<<<Mk_img, 256, 256*sizeof(float), stream>>>(
          kv_row_img_b, Mk_img, K, H, D,
          p.norm_added_k_gamma, 1.0e-6f,
          dKh_img_b, dVh_img_b);
      CUDA_CHECK(cudaGetLastError());
    }
    rec(EV_AFTER_IMG_KV_PACK);
  }
  if (prof_detail && !do_img) {
    rec(EV_AFTER_IMG_K_PROJ);
    rec(EV_AFTER_IMG_V_PROJ);
    rec(EV_AFTER_IMG_KV_PACK);
  }

  // 4) Run cuDNN cross-attention SDPA over concatenated [text; image]
  {
    auto handle = get_cudnn_handle();
    cudnnSetStream(handle, stream);

    GraphCacheKey cache_key{B, Mq, H, D, Mk_total};
    auto it = g_sdpa_graph_cache.find(cache_key);
    std::shared_ptr<fe::graph::Graph> graph;

    if (it != g_sdpa_graph_cache.end()) {
      graph = it->second;
    } else {
      auto t0 = std::chrono::high_resolution_clock::now();
      graph_built_now = true;
      graph = std::make_shared<fe::graph::Graph>();
      graph->set_io_data_type(fe::DataType_t::HALF)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      int64_t b = B;
      int64_t K_total = H * D;
      auto Q = graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("Q")
                                 .set_uid(1)
                                 .set_dim({b, H, Mq, D})
                                 .set_stride({int64_t(Mq) * K_total, D, K_total, 1}));

      auto K_t = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K")
                                   .set_uid(2)
                                   .set_dim({b, H, Mk_total, D})
                                   .set_stride({int64_t(Mk_total) * K_total, D, K_total, 1}));

      auto V_t = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V")
                                   .set_uid(3)
                                   .set_dim({b, H, Mk_total, D})
                                   .set_stride({int64_t(Mk_total) * K_total, D, K_total, 1}));

      float attn_scale = 1.0f / sqrtf(static_cast<float>(D));
      auto sdpa_options = fe::graph::SDPA_attributes()
                              .set_name("sdpa_cross_attention_i2v")
                              .set_attn_scale(attn_scale);

      auto [O, Stats] = graph->sdpa(Q, K_t, V_t, sdpa_options);

      O->set_output(true).set_dim({b, H, Mq, D}).set_stride({int64_t(Mq) * K_total, D, K_total, 1}).set_uid(4);

      if (!graph->build(handle, {fe::HeurMode_t::A}).is_good()) {
        return cudaErrorUnknown;
      }

      g_sdpa_graph_cache[cache_key] = graph;
      auto t1 = std::chrono::high_resolution_clock::now();
      graph_build_ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {1, dQh_bmhk}, {2, dKh_bmhk}, {3, dVh_bmhk}, {4, dOh_bmhk}};

    int64_t workspace_size = 0;
    if (!graph->get_workspace_size(workspace_size).is_good()) {
      return cudaErrorUnknown;
    }

    // Ensure cuDNN SDPA workspace exists with sufficient capacity.
    // Note: This is allocated separately from the monolithic workspace pool, so it can grow
    // without invalidating other pointers.
    p.workspace->ensure_cudnn_workspace((size_t)workspace_size);
    void* workspace = p.workspace->cudnn.cudnn_workspace;

    rec(EV_BEFORE_SDPA_EXEC);
    if (!graph->execute(handle, variant_pack, workspace).is_good()) {
      return cudaErrorUnknown;
    }
    rec(EV_AFTER_SDPA_EXEC);
  }

  // 5) Output projection (optionally fused with residual)
  {
    auto* fp8_scratch = reinterpret_cast<cutlass::half_t*>(dQh_bmhk); // safe scratch
    cudaError_t err = cudaSuccess;
    if (fuse_residual) {
      err = apply_linear_row_fused_residual<WeightT>(
        reinterpret_cast<const cutlass::half_t*>(dOh_bmhk),
        reinterpret_cast<const WeightT*>(p.w_out),
        reinterpret_cast<const cutlass::half_t*>(p.b_out),
        reinterpret_cast<cutlass::half_t*>(p.out_after_linear),
        M_total, K, K, stream,
        fp8_scratch,
        reinterpret_cast<const cutlass::half_t*>(p.w_out_scale),
        reinterpret_cast<cutlass::half_t*>(dQrow));
    } else {
      err = apply_linear_row<WeightT>(
        reinterpret_cast<const cutlass::half_t*>(dOh_bmhk),
        reinterpret_cast<const WeightT*>(p.w_out),
        reinterpret_cast<const cutlass::half_t*>(p.b_out),
        reinterpret_cast<cutlass::half_t*>(p.out_after_linear),
        M_total, K, K, stream,
        fp8_scratch,
        reinterpret_cast<const cutlass::half_t*>(p.w_out_scale));
    }
    if (err != cudaSuccess) { return err; }
  }
  rec(EV_AFTER_OUT_PROJ);

  if (prof_detail) {
    cudaEventSynchronize(ev[EV_AFTER_OUT_PROJ]);
    auto ms = [&](int a, int b) -> float {
      float out = 0.f;
      cudaEventElapsedTime(&out, ev[a], ev[b]);
      return out;
    };
    float t_q_proj   = ms(EV_START, EV_AFTER_Q_PROJ);
    float t_q_pack   = ms(EV_AFTER_Q_PROJ, EV_AFTER_Q_PACK);
    float t_kv_text  = ms(EV_AFTER_Q_PACK, EV_AFTER_TEXT_KV_PROJ);
    float t_pack_txt = ms(EV_AFTER_TEXT_KV_PROJ, EV_AFTER_TEXT_KV_PACK);
    float t_k_img    = ms(EV_AFTER_TEXT_KV_PACK, EV_AFTER_IMG_K_PROJ);
    float t_v_img    = ms(EV_AFTER_IMG_K_PROJ, EV_AFTER_IMG_V_PROJ);
    float t_pack_img = ms(EV_AFTER_IMG_V_PROJ, EV_AFTER_IMG_KV_PACK);
    float t_sdpa     = ms(EV_BEFORE_SDPA_EXEC, EV_AFTER_SDPA_EXEC);
    float t_out      = ms(EV_AFTER_SDPA_EXEC, EV_AFTER_OUT_PROJ);
    float t_total    = ms(EV_START, EV_AFTER_OUT_PROJ);

    std::printf(
      "[wan_ca_i2v_cudnn][call=%lld] B=%d Mq=%d K=%d H=%d D=%d Mk_text=%d Mk_img=%d | "
      "q_proj=%.3f q_pack=%.3f kv_text=%.3f pack_text=%.3f "
      "k_img=%.3f v_img=%.3f pack_img=%.3f sdpa=%.3f out=%.3f total=%.3f ms",
      prof_call_idx, B, Mq, K, H, D, Mk_text, Mk_img,
      t_q_proj, t_q_pack, t_kv_text, t_pack_txt,
      t_k_img, t_v_img, t_pack_img, t_sdpa, t_out, t_total
    );
    if (graph_built_now) {
      std::printf(" | graph_build_cpu=%.3f ms", graph_build_ms_cpu);
    }
    std::printf("\n");

    for (int i = 0; i < EV_COUNT; ++i) cudaEventDestroy(ev[i]);
  }

  return cudaSuccess;
}

template <typename WeightT>
static cudaError_t run_self_attention_cudnn(const AttentionDeviceParamsT<WeightT>& p, cudaStream_t stream) {
  return run_self_attention_cudnn_impl<WeightT>(p, stream, nullptr, nullptr);
}

// Sparge Attention Implementation
// ============================================================================

// Smooth-K support has been removed; K is used as-is without mean subtraction.

constexpr int kSpargeSm89CtaQ = 128;
constexpr int kSpargeSm89CtaK = 64;

// Helper: Convert cutlass::half_t to float
__device__ __forceinline__ float half_to_float(cutlass::half_t val) {
  return __half2float(*reinterpret_cast<const half*>(&val));
}

// Helper: Convert float to cutlass::half_t
__device__ __forceinline__ cutlass::half_t float_to_half(float val) {
  half h = __float2half(val);
  return *reinterpret_cast<cutlass::half_t*>(&h);
}

// Kernel: Mean pool blocks for block map computation (optimized with vectorized loads)
// Grid: (num_blocks, H), Block: (D/2) - each thread handles 2 elements via half2
// NOTE: Input X is in [M, H, D] layout (output of rmsnorm_pack_bmhk_rope_from_row_kernel)
//       Output X_pooled is in [num_blocks, H, D] layout (MHD-consistent, eliminates transpose)
// When subtract_mean=true, subtracts X_mean from each element before pooling
__global__ void sparge_mean_pool_kernel(
    const cutlass::half_t* __restrict__ X,       // [M, H, D] - stride [H*D, D, 1]
    const cutlass::half_t* __restrict__ X_mean,  // [H, D] - optional mean to subtract (nullptr if not used)
    cutlass::half_t* __restrict__ X_pooled,      // [num_blocks, H, D] - MHD layout
    int M, int H, int D, int BLOCK_SIZE,
    bool subtract_mean) {
  int block_idx = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;

  int start = block_idx * BLOCK_SIZE;
  int end = min(start + BLOCK_SIZE, M);
  int count = end - start;
  if (count <= 0) return;

  // Use half2 for vectorized access (2 elements per thread)
  const half2* X_h2 = reinterpret_cast<const half2*>(X);
  half2* out_h2 = reinterpret_cast<half2*>(X_pooled);
  int D2 = D / 2;
  int K2 = H * D2;  // Stride in half2 units for [*, H, D] layout

  // Load mean if needed (X_mean is [H, D], stored contiguously)
  float2 mean_val = make_float2(0.f, 0.f);
  if (subtract_mean && X_mean != nullptr && tid < D2) {
    const half2* mean_h2 = reinterpret_cast<const half2*>(X_mean);
    half2 m = mean_h2[h * D2 + tid];
    mean_val.x = __half2float(m.x);
    mean_val.y = __half2float(m.y);
  }

  if (tid < D2) {
    float2 sum = make_float2(0.f, 0.f);

    #pragma unroll 4
    for (int i = start; i < end; i++) {
      // [M, H, D] layout: X[m, h, d] at index m * H * D + h * D + d
      // In half2: X_h2[m * H * D2 + h * D2 + tid]
      half2 val = X_h2[i * K2 + h * D2 + tid];
      float2 fval;
      fval.x = __half2float(val.x) - mean_val.x;
      fval.y = __half2float(val.y) - mean_val.y;
      sum.x += fval.x;
      sum.y += fval.y;
    }

    float inv_count = 1.f / float(count);
    half2 result;
    result.x = __float2half(sum.x * inv_count);
    result.y = __float2half(sum.y * inv_count);

    // Output [num_blocks, H, D] layout: X_pooled[block_idx, h, d] at index block_idx * H * D + h * D + d
    // In half2: out_h2[block_idx * K2 + h * D2 + tid]
    out_h2[block_idx * K2 + h * D2 + tid] = result;
  }
}

__global__ void sparge_mean_pool_bf16_kernel(
    const cutlass::bfloat16_t* __restrict__ X,
    const cutlass::bfloat16_t* __restrict__ X_mean,
    cutlass::half_t* __restrict__ X_pooled,
    int M, int H, int D, int BLOCK_SIZE,
    bool subtract_mean) {
  int block_idx = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;

  int start = block_idx * BLOCK_SIZE;
  int end = min(start + BLOCK_SIZE, M);
  int count = end - start;
  if (count <= 0) return;

  const int K = H * D;
  const int pair_count = D / 2;
  if (tid < pair_count) {
    int d = tid * 2;

    float2 mean_val = make_float2(0.f, 0.f);
    if (subtract_mean && X_mean != nullptr) {
      mean_val = omnidreams_singleview::load_vec2_as_float2(X_mean + h * D + d);
    }

    float2 sum = make_float2(0.f, 0.f);
    for (int m = start; m < end; ++m) {
      float2 val = omnidreams_singleview::load_vec2_as_float2(X + size_t(m) * K + h * D + d);
      sum.x += val.x - mean_val.x;
      sum.y += val.y - mean_val.y;
    }

    float inv_count = 1.f / float(count);
    omnidreams_singleview::store_float2_as_vec2<cutlass::half_t>(
        X_pooled + (block_idx * H + h) * D + d,
        make_float2(sum.x * inv_count, sum.y * inv_count));
    return;
  }

  if ((D & 1) == 0 || tid != pair_count) return;

  int d = D - 1;
  float mean_val = 0.f;
  if (subtract_mean && X_mean != nullptr) {
    mean_val = omnidreams_singleview::to_float(X_mean[h * D + d]);
  }

  float sum = 0.f;
  for (int m = start; m < end; ++m) {
    sum += omnidreams_singleview::to_float(X[size_t(m) * K + h * D + d]) - mean_val;
  }

  X_pooled[(block_idx * H + h) * D + d] =
      omnidreams_singleview::from_float<cutlass::half_t>(sum / float(count));
}

// (Removed) K-mean kernel; smooth-K no longer used.

// Kernel: Compute pooled scores (Q_pooled @ K_pooled^T) - tiled for high throughput
// Grid: (num_q_blocks, H), Block: (128) - 4 warps independently process k_blocks
// Each warp computes full D-dimensional dot products using vectorized float2 loads.
// Q[q_blk, h, :] is loaded once into registers and reused across all k_blocks.
// Requires D == 128 (32 lanes x 4 elements per lane via float2 loads).
// NOTE: Inputs Q_pooled and K_pooled are in MHD layout [num_blocks, H, D]
//       Output scores is in [num_q_blocks, num_k_blocks, H] layout (MHD-consistent)
__global__ void sparge_pooled_score_kernel(
    const cutlass::half_t* __restrict__ Q_pooled,  // [num_q_blocks, H, D] - MHD layout
    const cutlass::half_t* __restrict__ K_pooled,  // [num_k_blocks, H, D] - MHD layout
    float* __restrict__ scores,                     // [num_q_blocks, num_k_blocks, H] - MHD-consistent
    int num_q_blocks, int num_k_blocks, int H, int D, float scale) {
  const int q_blk = blockIdx.x;
  const int h = blockIdx.y;
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int num_warps = blockDim.x / 32;

  // D=128: each lane loads 4 halfs as one float2 (8 bytes), 32 lanes cover all 128 elements
  const int D_f2 = D / 4;  // stride in float2 units (=32 for D=128)
  const float2* Q_f2 = reinterpret_cast<const float2*>(Q_pooled);
  const float2* K_f2 = reinterpret_cast<const float2*>(K_pooled);

  // Load Q[q_blk, h, :] into registers once - reused across all k_blocks
  const int q_off = (q_blk * H + h) * D_f2;
  float2 q_vec = Q_f2[q_off + lane_id];
  const half* qh = reinterpret_cast<const half*>(&q_vec);
  const float q0 = __half2float(qh[0]);
  const float q1 = __half2float(qh[1]);
  const float q2 = __half2float(qh[2]);
  const float q3 = __half2float(qh[3]);

  // Output base offset for this (q_blk, h): scores[q_blk, k, h]
  const int score_base = q_blk * num_k_blocks * H + h;

  // Each warp independently processes k_blocks in round-robin
  for (int k = warp_id; k < num_k_blocks; k += num_warps) {
    const int k_off = (k * H + h) * D_f2;
    float2 k_vec = K_f2[k_off + lane_id];
    const half* kh = reinterpret_cast<const half*>(&k_vec);

    float sum = q0 * __half2float(kh[0])
              + q1 * __half2float(kh[1])
              + q2 * __half2float(kh[2])
              + q3 * __half2float(kh[3]);

    // Warp-level reduction (5 steps for 32 lanes)
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane_id == 0)
      scores[score_base + k * H] = sum * scale;
  }
}

// Kernel: TopK selection and sparse map creation - parallel version
// Grid: (num_q_blocks, H), Block: (256)
// Uses parallel reduction for finding max, then marks and repeats
// Shared memory layout: float[256] s_max, int[256] s_idx, int8_t[num_k_blocks] s_used
// NOTE: scores is in [num_q_blocks, num_k_blocks, H] layout (MHD-consistent)
//       sparse_map and topk_indices outputs are in HMD layout for SpargeAttn
__global__ void sparge_topk_sparse_map_kernel(
    const float* __restrict__ scores,     // [num_q_blocks, num_k_blocks, H] - MHD-consistent
    int32_t* __restrict__ sparse_map,     // [H, num_q_blocks, num_k_blocks] - HMD layout for SpargeAttn
    int32_t* __restrict__ topk_indices,   // [H, num_q_blocks, topk] - HMD layout for SpargeAttn
    int32_t* __restrict__ lut,             // [H, num_q_blocks, num_k_blocks] - HMD for SpargeAttn (merged from build_lut)
    int32_t* __restrict__ valid_block_num, // [H, num_q_blocks] - HMD for SpargeAttn (merged from build_lut)
    int num_q_blocks, int num_k_blocks, int H, int topk,
    int q_to_k_token_offset,
    bool attention_sink) {
  int q_blk = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;

  // Scores input: [num_q_blocks, num_k_blocks, H] -> stride = [num_k_blocks * H, H, 1]
  int score_base = q_blk * num_k_blocks * H + h;
  int score_stride = H;  // k_blk stride in scores

  // Sparse map output: [H, num_q_blocks, num_k_blocks] -> offset = h * num_q_blocks * num_k_blocks + q_blk * num_k_blocks
  // NOTE: HMD layout required for SpargeAttn compatibility
  int sparse_offset = h * num_q_blocks * num_k_blocks + q_blk * num_k_blocks;

  // TopK output: [H, num_q_blocks, topk] -> offset = h * num_q_blocks * topk + q_blk * topk
  int topk_offset = h * num_q_blocks * topk + q_blk * topk;

  // Dynamic shared memory layout
  extern __shared__ char smem_raw[];
  float* s_max = reinterpret_cast<float*>(smem_raw);
  int* s_idx = reinterpret_cast<int*>(smem_raw + 256 * sizeof(float));
  int8_t* s_used = reinterpret_cast<int8_t*>(smem_raw + 256 * sizeof(float) + 256 * sizeof(int));

  // Initialize sparse map to 0 and copy to shared
  for (int k = tid; k < num_k_blocks; k += blockDim.x) {
    sparse_map[sparse_offset + k] = 0;
    s_used[k] = 0;
  }
  __syncthreads();

  // Iterate topk times to find top-k elements
  for (int t = 0; t < topk && t < num_k_blocks; t++) {
    // Each thread finds max in its assigned elements
    float thread_max = -1e30f;
    int thread_idx = -1;

    for (int k = tid; k < num_k_blocks; k += blockDim.x) {
      if (s_used[k] == 0) {
        // Score at [q_blk, k, h] = score_base + k * score_stride
        float val = scores[score_base + k * score_stride];
        if (val > thread_max) {
          thread_max = val;
          thread_idx = k;
        }
      }
    }

    s_max[tid] = thread_max;
    s_idx[tid] = thread_idx;
    __syncthreads();

    // Parallel reduction to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        if (s_max[tid + s] > s_max[tid]) {
          s_max[tid] = s_max[tid + s];
          s_idx[tid] = s_idx[tid + s];
        }
      }
      __syncthreads();
    }

    // Thread 0 records the result
    if (tid == 0) {
      int best_idx = s_idx[0];
      if (best_idx >= 0) {
        sparse_map[sparse_offset + best_idx] = 1;
        topk_indices[topk_offset + t] = best_idx;
        s_used[best_idx] = 1;  // Mark as used for next iteration
      }
    }
    __syncthreads();
  }

  // Build delta-encoded LUT (merged from sparge_build_lut_kernel)
  // Thread 0 scans sparse_map and writes delta-encoded block indices
  if (tid == 0) {
    if (attention_sink && num_k_blocks > 0) {
      sparse_map[sparse_offset] = 1;
    }
    if (q_to_k_token_offset >= 0 && num_k_blocks > 0) {
      int q_token_start = q_blk * kSpargeSm89CtaQ;
      int q_token_end = q_token_start + kSpargeSm89CtaQ;
      int k_token_start = q_to_k_token_offset + q_token_start;
      int k_token_end = q_to_k_token_offset + q_token_end - 1;
      int local_k_start = max(0, min(num_k_blocks - 1, k_token_start / kSpargeSm89CtaK));
      int local_k_end = max(0, min(num_k_blocks - 1, k_token_end / kSpargeSm89CtaK));
      for (int k = local_k_start; k <= local_k_end; ++k) {
        sparse_map[sparse_offset + k] = 1;
      }
    }
    if (lut != nullptr && valid_block_num != nullptr) {
      int lut_offset = sparse_offset;  // same HMD layout
      int valid_offset = h * num_q_blocks + q_blk;
      int count = 0;
      int prev_block = 0;
      for (int k = 0; k < num_k_blocks; k++) {
        if (sparse_map[sparse_offset + k]) {
          lut[lut_offset + count] = k - prev_block;
          prev_block = k;
          count++;
        }
      }
      valid_block_num[valid_offset] = count;
    }
  }
}

// Dense Sparge map for topk=100% coverage.
//
// The SpargeAttn kernel consumes a delta-encoded LUT in HMD layout. For the
// dense Sparge path every K block is valid, so the LUT is simply [0, 1, 1, ...]
// for each (head, q_block). This avoids the Sparge pooled-score/top-k prep
// launches when the caller requested dense coverage.
__global__ void sparge_dense_lut_kernel(
    int32_t* __restrict__ lut,
    int32_t* __restrict__ valid_block_num,
    int num_q_blocks,
    int num_k_blocks,
    int H) {
  int q_blk = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;
  if (q_blk >= num_q_blocks || h >= H) return;

  int lut_offset = h * num_q_blocks * num_k_blocks + q_blk * num_k_blocks;
  for (int k = tid; k < num_k_blocks; k += blockDim.x) {
    lut[lut_offset + k] = (k == 0) ? 0 : 1;
  }
  if (tid == 0) {
    valid_block_num[h * num_q_blocks + q_blk] = num_k_blocks;
  }
}

// Kernel: Convert sparse map to delta-encoded LUT
// Grid: (num_q_blocks, H), Block: (1)
// NOTE: sparse_map is in [H, num_q_blocks, num_k_blocks] layout (HMD)
//       lut output is in [H, num_q_blocks, num_k_blocks] layout (HMD)
//       valid_block_num is in [H, num_q_blocks] layout (HMD)
__global__ void sparge_build_lut_kernel(
    const int32_t* __restrict__ sparse_map,  // [H, num_q_blocks, num_k_blocks] - HMD for SpargeAttn
    int32_t* __restrict__ lut,               // [H, num_q_blocks, num_k_blocks] - HMD for SpargeAttn
    int32_t* __restrict__ valid_block_num,   // [H, num_q_blocks] - HMD for SpargeAttn
    int num_q_blocks, int H, int num_k_blocks, int topk) {
  int q_blk = blockIdx.x;
  int h = blockIdx.y;

  // HMD layout: [H, num_q_blocks, num_k_blocks] -> offset = h * num_q_blocks * num_k_blocks + q_blk * num_k_blocks
  // NOTE: HMD layout required for SpargeAttn compatibility
  int offset = h * num_q_blocks * num_k_blocks + q_blk * num_k_blocks;
  int lut_offset = offset;
  // valid_block_num: [H, num_q_blocks] -> offset = h * num_q_blocks + q_blk
  int valid_offset = h * num_q_blocks + q_blk;

  int count = 0;
  int prev_block = 0;

  for (int k = 0; k < num_k_blocks && count < topk; k++) {
    if (sparse_map[offset + k]) {
      lut[lut_offset + count] = k - prev_block;
      prev_block = k;
      count++;
    }
  }
  valid_block_num[valid_offset] = count;
}

// Kernel: Per-block quantization of Q/K to int8 with zero-padding
// Grid: (num_blocks, H), Block: (BLOCK_SIZE)
// NOTE: Input X is in [M, H, D] layout (output of rmsnorm_pack_bmhk_rope_from_row_kernel)
//       Output X_int8 is also in [M, H, D] layout (for SpargeAttn compatibility)
//       For the last block, positions beyond M are zero-padded to avoid out-of-bounds reads
__global__ void sparge_quantize_kernel(
    const cutlass::half_t* __restrict__ X,  // [M, H, D] - stride [H*D, D, 1]
    const cutlass::half_t* __restrict__ X_mean,  // [H, D] or nullptr
    int8_t* __restrict__ X_int8,             // [padded_M, H, D] - stride [H*D, D, 1]
    float* __restrict__ X_scale,             // [H, num_blocks]
    int M, int H, int D, int BLOCK_SIZE, bool subtract_mean) {
  int block_idx = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;

  int start = block_idx * BLOCK_SIZE;
  int block_end = start + BLOCK_SIZE;  // End of this block (may exceed M)
  int valid_end = min(block_end, M);   // End of valid data
  if (start >= M) {
    // Block is entirely in padding region - write zeros
    for (int i = start + (tid / D); i < block_end; i += (blockDim.x / D)) {
      int d = tid % D;
      if (d < D) {
        X_int8[i * H * D + h * D + d] = 0;
      }
    }
    return;
  }

  extern __shared__ float smem[];
  float* smax = smem;

  // Compute stride for [M, H, D] layout
  int K = H * D;  // Total hidden dimension

  // Find max abs value in valid portion of block
  float local_max = 0.f;
  for (int i = start + (tid / D); i < valid_end; i += (blockDim.x / D)) {
    int d = tid % D;
    if (d < D && i < valid_end) {
      // [M, H, D] layout: X[m, h, d] = X[m * H * D + h * D + d]
      float val = half_to_float(X[i * K + h * D + d]);
      if (subtract_mean && X_mean != nullptr) {
        val -= half_to_float(X_mean[h * D + d]);
      }
      local_max = fmaxf(local_max, fabsf(val));
    }
  }

  // Block reduce to find max
  smax[tid] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smax[tid] = fmaxf(smax[tid], smax[tid + s]);
    }
    __syncthreads();
  }

  float scale = smax[0] / 127.f + 1e-7f;
  if (tid == 0) {
    X_scale[h * ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) + block_idx] = scale;
  }
  __syncthreads();

  // Quantize valid data
  for (int i = start + (tid / D); i < valid_end; i += (blockDim.x / D)) {
    int d = tid % D;
    if (d < D && i < valid_end) {
      // [M, H, D] layout: X[m, h, d] = X[m * H * D + h * D + d]
      float val = half_to_float(X[i * K + h * D + d]);
      if (subtract_mean && X_mean != nullptr) {
        val -= half_to_float(X_mean[h * D + d]);
      }
      int q = __float2int_rn(val / scale);
      q = max(-127, min(127, q));
      X_int8[i * K + h * D + d] = static_cast<int8_t>(q);
    }
  }

  // Zero-pad positions beyond M (last block only)
  for (int i = valid_end + (tid / D); i < block_end; i += (blockDim.x / D)) {
    int d = tid % D;
    if (d < D) {
      X_int8[i * K + h * D + d] = 0;
    }
  }
}

__global__ void sparge_quantize_bf16_kernel(
    const cutlass::bfloat16_t* __restrict__ X,
    const cutlass::bfloat16_t* __restrict__ X_mean,
    int8_t* __restrict__ X_int8,
    float* __restrict__ X_scale,
    int M, int H, int D, int BLOCK_SIZE, bool subtract_mean) {
  int block_idx = blockIdx.x;
  int h = blockIdx.y;
  int tid = threadIdx.x;

  int start = block_idx * BLOCK_SIZE;
  int block_end = start + BLOCK_SIZE;
  int valid_end = min(block_end, M);
  if (start >= M) {
    for (int i = start + (tid / D); i < block_end; i += (blockDim.x / D)) {
      int d = tid % D;
      if (d < D) {
        X_int8[i * H * D + h * D + d] = 0;
      }
    }
    return;
  }

  extern __shared__ float smem[];
  float* smax = smem;
  int K = H * D;

  float local_max = 0.f;
  for (int i = start + (tid / D); i < valid_end; i += (blockDim.x / D)) {
    int d = tid % D;
    if (d < D && i < valid_end) {
      float val = omnidreams_singleview::to_float(X[i * K + h * D + d]);
      if (subtract_mean && X_mean != nullptr) {
        val -= omnidreams_singleview::to_float(X_mean[h * D + d]);
      }
      local_max = fmaxf(local_max, fabsf(val));
    }
  }

  smax[tid] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smax[tid] = fmaxf(smax[tid], smax[tid + s]);
    }
    __syncthreads();
  }

  float scale = smax[0] / 127.f + 1e-7f;
  if (tid == 0) {
    X_scale[h * ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) + block_idx] = scale;
  }
  __syncthreads();

  for (int i = start + (tid / D); i < valid_end; i += (blockDim.x / D)) {
    int d = tid % D;
    if (d < D && i < valid_end) {
      float val = omnidreams_singleview::to_float(X[i * K + h * D + d]);
      if (subtract_mean && X_mean != nullptr) {
        val -= omnidreams_singleview::to_float(X_mean[h * D + d]);
      }
      int q = __float2int_rn(val / scale);
      q = max(-127, min(127, q));
      X_int8[i * K + h * D + d] = static_cast<int8_t>(q);
    }
  }

  for (int i = valid_end + (tid / D); i < block_end; i += (blockDim.x / D)) {
    int d = tid % D;
    if (d < D) {
      X_int8[i * K + h * D + d] = 0;
    }
  }
}

// Helper: Compute SpargeAttn permutation for V tensor
// This permutes elements within 16-element blocks for optimized fp8 matmul access
// Pattern: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] -> [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15]
__device__ __forceinline__ int sparge_v_permute(int m) {
  int base = (m / 16) * 16;
  int mod = m % 16;
  // Formula from SpargeAttn's TransposePadPermuteKernel
  return base + (mod / 8) * 2 + ((mod / 2) % 4) * 4 + (mod % 2);
}

// Kernel A: Compute per-(h,d) absmax of V for FP8 quantization
// Grid: (H, num_tiles), Block: (D) where D=128
// Each block handles one head, one tile of V_ABSMAX_TILE_M rows.
// Threads map 1:1 to D values -> coalesced reads within each row.
// Uses atomicMax on positive floats to accumulate across tiles.
// V_scale must be zeroed before launch.
constexpr int V_ABSMAX_TILE_M = 256;

__global__ void sparge_v_absmax_kernel(
    const cutlass::half_t* __restrict__ V,  // [M, H, D] - stride [H*D, D, 1]
    float* __restrict__ V_scale,            // [H, D] - must be zeroed before launch
    int M, int H, int D) {
  int h = blockIdx.x;
  int tile = blockIdx.y;
  int d = threadIdx.x;  // 0..D-1, coalesced along D
  if (d >= D) return;
  int K = H * D;

  int m_start = tile * V_ABSMAX_TILE_M;
  int m_end = min(m_start + V_ABSMAX_TILE_M, M);

  // Accumulate per-d absmax over this tile
  float local_max = 0.f;
  for (int m = m_start; m < m_end; m++) {
    float val = fabsf(half_to_float(V[m * K + h * D + d]));
    local_max = fmaxf(local_max, val);
  }

  // Atomic max into global V_scale (positive floats sort like unsigned ints)
  atomicMax(reinterpret_cast<int*>(&V_scale[h * D + d]), __float_as_int(local_max));
}

__global__ void sparge_v_absmax_bf16_kernel(
    const cutlass::bfloat16_t* __restrict__ V,
    float* __restrict__ V_scale,
    int M, int H, int D) {
  int h = blockIdx.x;
  int tile = blockIdx.y;
  int d = threadIdx.x;
  if (d >= D) return;
  int K = H * D;

  int m_start = tile * V_ABSMAX_TILE_M;
  int m_end = min(m_start + V_ABSMAX_TILE_M, M);

  float local_max = 0.f;
  for (int m = m_start; m < m_end; m++) {
    float val = fabsf(omnidreams_singleview::to_float(V[m * K + h * D + d]));
    local_max = fmaxf(local_max, val);
  }

  atomicMax(reinterpret_cast<int*>(&V_scale[h * D + d]), __float_as_int(local_max));
}

// Kernel A': Finalize V_scale: apply headroom factor
// Grid: ceil(total/256), Block: 256
__global__ void sparge_v_scale_finalize_kernel(
    float* __restrict__ V_scale, int total) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    V_scale[idx] = V_scale[idx] / 2.25f + 1e-7f;
  }
}

// Kernel B: Quantize V to FP8 with transposition via shared memory
// Grid: (H, ceil(padded_M / TILE_M)), Block: (256)
// Uses smem to transpose V[M,H,D] -> V_fp8[H,D,padded_M] with SpargeAttn permutation.
// TILE_M=128: each block processes a 128xD tile of V for one head.
// Shared memory: half smem[TILE_M][D+2] = 128*130*2 = 33,280 bytes (pad to avoid bank conflicts).
constexpr int V_QUANT_TILE_M = 128;
constexpr int V_QUANT_SMEM_PAD = 2;  // padding to avoid bank conflicts

__global__ void sparge_v_quantize_permute_kernel(
    const cutlass::half_t* __restrict__ V,     // [M, H, D] - stride [H*D, D, 1]
    const float* __restrict__ V_scale,         // [H, D]
    cutlass::float_e4m3_t* __restrict__ V_fp8, // [H, D, padded_M]
    int M, int H, int D, int padded_M) {
  int h = blockIdx.x;
  int tile_idx = blockIdx.y;
  int tid = threadIdx.x;
  int K = H * D;
  int m_base = tile_idx * V_QUANT_TILE_M;

  // Shared memory for the tile: [TILE_M][D + pad]
  extern __shared__ cutlass::half_t v_smem[];
  int smem_stride = D + V_QUANT_SMEM_PAD;

  // Phase 1: Load V tile into smem (coalesced reads along D dimension)
  // 256 threads, D=128: 2 rows per iteration, need TILE_M/2 = 64 iterations
  int rows_per_iter = 256 / D;  // = 2 for D=128
  for (int iter = 0; iter < (V_QUANT_TILE_M + rows_per_iter - 1) / rows_per_iter; iter++) {
    int local_row = iter * rows_per_iter + tid / D;
    int d = tid % D;
    int m = m_base + local_row;
    cutlass::half_t val;
    if (local_row < V_QUANT_TILE_M && m < M) {
      val = V[m * K + h * D + d];
    } else {
      val = cutlass::half_t(0);  // pad with zero
    }
    if (local_row < V_QUANT_TILE_M) {
      v_smem[local_row * smem_stride + d] = val;
    }
  }
  __syncthreads();

  // Phase 2: Quantize + permute + write V_fp8 (coalesced writes along M dimension)
  // Remap threads: tid % TILE_M gives m_local, tid / TILE_M gives d_offset (0 or 1)
  // 256 threads / 128 TILE_M = 2 d-values per iteration
  int m_local = tid % V_QUANT_TILE_M;
  int d_off = tid / V_QUANT_TILE_M;  // 0 or 1
  int m_global = m_base + m_local;
  int permuted_m = sparge_v_permute(m_global);

  // Iterate over D dimension: 2 d-values per iteration, D/2 = 64 iterations
  for (int iter = 0; iter < (D + 1) / 2; iter++) {
    int d = iter * 2 + d_off;
    if (d < D) {
      float scale = V_scale[h * D + d];
      float val = half_to_float(v_smem[m_local * smem_stride + d]);
      val = fmaxf(-448.f, fminf(448.f, val / scale));
      V_fp8[h * D * padded_M + d * padded_M + permuted_m] =
          cutlass::float_e4m3_t(val);
    }
  }
}

__global__ void sparge_v_quantize_permute_bf16_kernel(
    const cutlass::bfloat16_t* __restrict__ V,
    const float* __restrict__ V_scale,
    cutlass::float_e4m3_t* __restrict__ V_fp8,
    int M, int H, int D, int padded_M) {
  int h = blockIdx.x;
  int tile_idx = blockIdx.y;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int K = H * D;
  int m_base = tile_idx * V_QUANT_TILE_M;

  constexpr int kWarps = 8;
  for (int m_off = 0; m_off < V_QUANT_TILE_M; m_off += 32) {
    int m = m_base + m_off + lane;
    int permuted_m = sparge_v_permute(m);
    for (int d_base = 0; d_base < D; d_base += kWarps) {
      int d = d_base + warp;
      if (d < D) {
        float val = 0.0f;
        if (m < M) {
          val = omnidreams_singleview::to_float(V[m * K + h * D + d]);
        }
        float scale = V_scale[h * D + d];
        val = fmaxf(-448.f, fminf(448.f, val / scale));
        V_fp8[h * D * padded_M + d * padded_M + permuted_m] =
            cutlass::float_e4m3_t(val);
      }
    }
  }
}

__global__ void cosmos_sparge_bf16_to_half_kernel(
    const cutlass::bfloat16_t* __restrict__ src,
    cutlass::half_t* __restrict__ dst,
    int64_t n) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = omnidreams_singleview::from_float<cutlass::half_t>(omnidreams_singleview::to_float(src[idx]));
}

__global__ void cosmos_sparge_half_to_bf16_kernel(
    const cutlass::half_t* __restrict__ src,
    cutlass::bfloat16_t* __restrict__ dst,
    int64_t n) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  dst[idx] = omnidreams_singleview::from_float<cutlass::bfloat16_t>(half_to_float(src[idx]));
}

__global__ void cosmos_sparge_fill_one_float_kernel(float* dst, float value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dst[0] = value;
  }
}

__global__ void cosmos_sparge_fill_float_kernel(float* dst, float value, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = value;
  }
}

static cudaError_t cosmos_sparge_convert_bf16_to_half(
    const cutlass::bfloat16_t* src,
    cutlass::half_t* dst,
    int64_t n,
    cudaStream_t stream) {
  if (!src || !dst || n <= 0) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cosmos_sparge_bf16_to_half_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      src, dst, n);
  return cudaGetLastError();
}

static cudaError_t cosmos_sparge_convert_half_to_bf16(
    const cutlass::half_t* src,
    cutlass::bfloat16_t* dst,
    int64_t n,
    cudaStream_t stream) {
  if (!src || !dst || n <= 0) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  cosmos_sparge_half_to_bf16_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
      src, dst, n);
  return cudaGetLastError();
}

cudaError_t run_cosmos_sparge_attention_bf16(
    const cutlass::bfloat16_t* Q,
    const cutlass::bfloat16_t* K,
    const cutlass::bfloat16_t* V,
    cutlass::bfloat16_t* O,
    ts::SpargeAttentionWorkspace* workspace,
    cutlass::half_t* qkv_half_scratch,
    int Mq,
    int Mk,
    int H,
    int D,
    float topk_ratio,
    bool attention_sink,
    cudaStream_t stream) {
#ifndef OMNIDREAMS_SINGLEVIEW_HAS_SPARGE
  (void)Q;
  (void)K;
  (void)V;
  (void)O;
  (void)workspace;
  (void)qkv_half_scratch;
  (void)Mq;
  (void)Mk;
  (void)H;
  (void)D;
  (void)topk_ratio;
  (void)attention_sink;
  (void)stream;
  return cudaErrorNotSupported;
#else
  if (!Q || !K || !V || !O || !workspace || !qkv_half_scratch) {
    return cudaErrorInvalidValue;
  }
  if (Mq <= 0 || Mk <= 0 || H <= 0 || D != 128) {
    return cudaErrorNotSupported;
  }
  if (!(topk_ratio > 0.0f && topk_ratio <= 1.0f)) {
    return cudaErrorInvalidValue;
  }
  const int K_total = H * D;
  const int64_t q_elems = int64_t(Mq) * K_total;
  cutlass::half_t* qh = qkv_half_scratch;
  ts::SpargeAttentionWorkspace& ws = *workspace;

  cudaError_t err = cosmos_sparge_convert_bf16_to_half(Q, qh, q_elems, stream);
  if (err != cudaSuccess) return err;

  const int BLKQ = ws.BLKQ;
  const int BLKK = ws.BLKK;
  if (BLKQ != kSpargeSm89CtaQ || BLKK != kSpargeSm89CtaK) {
    return cudaErrorNotSupported;
  }
  const int num_q_blocks = (Mq + BLKQ - 1) / BLKQ;
  const int num_k_blocks = (Mk + BLKK - 1) / BLKK;
  const int num_q_scale_blocks = num_q_blocks;
  const int num_k_scale_blocks = num_k_blocks;
  const int topk_raw = static_cast<int>(topk_ratio * static_cast<float>(num_k_blocks));
  const int topk = max(1, min(num_k_blocks, topk_raw));
  const int allocated_topk_raw =
      static_cast<int>(ws.topk_ratio * static_cast<float>(num_k_blocks));
  const int allocated_topk = max(1, min(num_k_blocks, allocated_topk_raw));
  if (topk > allocated_topk) {
    return cudaErrorInvalidValue;
  }
  const bool dense_sparge_map = topk >= num_k_blocks;
  const int padded_kv_len = ((Mk + 127) / 128) * 128;

  const float scale = 1.0f / sqrtf(static_cast<float>(D));
  if (dense_sparge_map) {
    sparge_dense_lut_kernel<<<dim3(num_q_blocks, H), 128, 0, stream>>>(
        ws.lut, ws.valid_block_num, num_q_blocks, num_k_blocks, H);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  } else {
    sparge_mean_pool_kernel<<<dim3(num_q_blocks, H), D / 2, 0, stream>>>(
        qh, nullptr, ws.pooled_q, Mq, H, D, BLKQ, false);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    int k_pool_threads = (D + 1) / 2;
    sparge_mean_pool_bf16_kernel<<<dim3(num_k_blocks, H), k_pool_threads, 0, stream>>>(
        K, nullptr, ws.pooled_k, Mk, H, D, BLKK, false);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    sparge_pooled_score_kernel<<<dim3(num_q_blocks, H), 128, 0, stream>>>(
        ws.pooled_q, ws.pooled_k, ws.pooled_scores,
        num_q_blocks, num_k_blocks, H, D, scale);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    size_t smem_topk = 256 * sizeof(float) + 256 * sizeof(int) +
                       size_t(num_k_blocks) * sizeof(int8_t);
    sparge_topk_sparse_map_kernel<<<dim3(num_q_blocks, H), 256, smem_topk, stream>>>(
        ws.pooled_scores, ws.sparse_map, ws.topk_indices,
        ws.lut, ws.valid_block_num,
        num_q_blocks, num_k_blocks, H, topk, max(0, Mk - Mq), attention_sink);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }

  int q_block_threads = min(256, BLKQ * D);
  sparge_quantize_kernel<<<dim3(num_q_scale_blocks, H), q_block_threads,
      q_block_threads * sizeof(float), stream>>>(
      qh, nullptr, ws.q_int8, ws.q_scale, Mq, H, D, BLKQ, false);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int k_block_threads = min(256, BLKK * D);
  sparge_quantize_bf16_kernel<<<dim3(num_k_scale_blocks, H), k_block_threads,
      k_block_threads * sizeof(float), stream>>>(
      K, nullptr, ws.k_int8, ws.k_scale, Mk, H, D, BLKK, false);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  err = cudaMemsetAsync(ws.v_scale, 0, size_t(H) * D * sizeof(float), stream);
  if (err != cudaSuccess) return err;
  int absmax_tiles = (Mk + V_ABSMAX_TILE_M - 1) / V_ABSMAX_TILE_M;
  sparge_v_absmax_bf16_kernel<<<dim3(H, absmax_tiles), D, 0, stream>>>(
      V, ws.v_scale, Mk, H, D);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  sparge_v_scale_finalize_kernel<<<(H * D + 255) / 256, 256, 0, stream>>>(
      ws.v_scale, H * D);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  int num_tiles = (padded_kv_len + V_QUANT_TILE_M - 1) / V_QUANT_TILE_M;
  sparge_v_quantize_permute_bf16_kernel<<<dim3(H, num_tiles), 256, 0, stream>>>(
      V, ws.v_scale, ws.v_fp8, Mk, H, D, padded_kv_len);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;

  cosmos_sparge_fill_float_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
      ws.pooled_scores, 1.0e6f, H);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  err = sparge_attention::run_sparge_attention_sm89_hd128(
      ws.q_int8, ws.k_int8,
      reinterpret_cast<__nv_fp8_e4m3*>(ws.v_fp8),
      reinterpret_cast<half*>(ws.o_sparse),
      ws.lut, ws.valid_block_num, ws.pooled_scores,
      ws.q_scale, ws.k_scale, ws.v_scale,
      1, Mq, Mk, H, H, padded_kv_len, scale, stream);
  if (err != cudaSuccess) return err;

  err = cosmos_sparge_convert_half_to_bf16(ws.o_sparse, O, q_elems, stream);
  if (err != cudaSuccess) return err;
  return cudaSuccess;
#endif
}


// ============================================================================
// Public API - Backend Dispatcher
// ============================================================================

template <typename WeightT>
cudaError_t run_self_attention(const AttentionDeviceParamsT<WeightT>& p, cudaStream_t stream) {
    switch (g_attention_backend) {
        case AttnBackend::CUTLASS_FLASH:
            if constexpr (std::is_same<WeightT, cutlass::half_t>::value) {
              return run_self_attention_cutlass(p, stream);
            }
            else {
              return cudaErrorNotSupported;
            }
        case AttnBackend::CUDNN:
            return run_self_attention_cudnn_impl<WeightT>(p, stream, nullptr, nullptr);
        default:
            return cudaErrorNotSupported;
    }
}

template <typename WeightT>
cudaError_t run_cross_attention(const AttentionDeviceParamsT<WeightT>& p, bool fuse_residual, cudaStream_t stream) {
    switch (g_attention_backend) {
        case AttnBackend::CUTLASS_FLASH:
            if constexpr (std::is_same<WeightT, cutlass::half_t>::value) {
              return run_cross_attention_cutlass(p, fuse_residual, stream);
            }
            else {
              return cudaErrorNotSupported;
            }
        case AttnBackend::CUDNN:
            return run_cross_attention_cudnn<WeightT>(p, fuse_residual, stream);
        default:
            return cudaErrorNotSupported;
    }
}

template <typename WeightT>
cudaError_t run_cross_attention_i2v(const CrossAttentionI2VParamsT<WeightT>& p, bool fuse_residual, cudaStream_t stream) {
    switch (g_attention_backend) {
    case AttnBackend::CUTLASS_FLASH:
        if constexpr (std::is_same<WeightT, cutlass::half_t>::value) {
            return run_cross_attention_i2v_cutlass(p, fuse_residual, stream);
        } else {
            return cudaErrorNotSupported;
        }
    case AttnBackend::CUDNN:
        return run_cross_attention_i2v_cudnn<WeightT>(p, fuse_residual, stream);
    default:
        return cudaErrorNotSupported;
    }
}

// Explicit template instantiations for linkage (mirrors run_cross_attention)
template cudaError_t run_cross_attention_i2v<cutlass::half_t>(
    const CrossAttentionI2VParamsT<cutlass::half_t>& p,
    bool fuse_residual,
    cudaStream_t stream
);
template cudaError_t run_cross_attention_i2v<cutlass::float_e4m3_t>(
    const CrossAttentionI2VParamsT<cutlass::float_e4m3_t>& p,
    bool fuse_residual,
    cudaStream_t stream
);
template cudaError_t run_cross_attention_i2v<int8_t>(
    const CrossAttentionI2VParamsT<int8_t>& p,
    bool fuse_residual,
    cudaStream_t stream
);
// Explicit template instantiations to ensure linkage across translation units
template cudaError_t run_self_attention<cutlass::half_t>(
  const AttentionDeviceParamsT<cutlass::half_t>& p,
  cudaStream_t stream
);
template cudaError_t run_cross_attention<cutlass::half_t>(
  const AttentionDeviceParamsT<cutlass::half_t>& p,
  bool fuse_residual,
  cudaStream_t stream
);
template cudaError_t run_self_attention<cutlass::float_e4m3_t>(
  const AttentionDeviceParamsT<cutlass::float_e4m3_t>& p,
  cudaStream_t stream
);
template cudaError_t run_cross_attention<cutlass::float_e4m3_t>(
  const AttentionDeviceParamsT<cutlass::float_e4m3_t>& p,
  bool fuse_residual,
  cudaStream_t stream
);
template cudaError_t run_self_attention<int8_t>(
  const AttentionDeviceParamsT<int8_t>& p,
  cudaStream_t stream
);
template cudaError_t run_cross_attention<int8_t>(
  const AttentionDeviceParamsT<int8_t>& p,
  bool fuse_residual,
  cudaStream_t stream
);

} // namespace omnidreams_singleview
