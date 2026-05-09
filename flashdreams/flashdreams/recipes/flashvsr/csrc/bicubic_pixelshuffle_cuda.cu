// Fused bicubic-upres + temporal replicate-pad-left + spatial pixel-shuffle
// CUDA extension for the FlashVSR encoder.
//
// One kernel folds five PyTorch-eager ops into one launch:
//   1. F.pad along T axis with replicate-left (cold-start only; folded as
//      ``t_in = max(0, t_padded - n_left_padding)`` in the index math).
//   2. permute+reshape from BCTHW -> (B*T)CHW (was a full-input copy).
//   3. F.interpolate(mode="bicubic", align_corners=False).
//   4. view+permute back to BCTHW.
//   5. PixelShuffle3d "channel_first" rearrange
//      ``b c (h hh) (w ww) -> b (c hh ww) f h w`` (ff=1, hh=ww=16).
//
// Two outputs in one pass:
//   - ``proj_input``: ``[B, 3*256, target_T, target_H/16, target_W/16]``
//     directly in the projector's conv1 input layout.
//   - ``last_upres``: ``[B, 3, T_raw, target_H, target_W]``, the un-padded
//     full-resolution upres consumed by the decoder/color-corrector.
// Each ``last_upres`` element is hit by exactly one thread (the
// pixel-shuffle index is bijective in (c, hh_idx, ww_idx, h_ps, w_ps) <->
// (c, y_full, x_full)), so no atomics are required.
//
// Bicubic math mirrors ATen's
// ``aten/src/ATen/native/cuda/UpSampleBicubic2d.cu`` (cubic-convolution
// with A = -0.75, separable 4x4 with replicate-clamp boundary, fp32
// accumulation regardless of input dtype). Output stored back at the
// input's dtype (bf16 / fp16 / fp32).
//
// Intentionally simpler than ``color_corrector_adain_cuda.cu``: no
// cooperative launch, no L2 access-policy hints, no vectorised
// loads/stores, no strided fallback. The bicubic 4x4 access pattern
// doesn't align cleanly to ``__nv_bfloat162``, and the upstream profile
// has this stage at ~0.6% of per-AR-step latency; the cheap path is
// enough.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <tuple>

namespace {

constexpr int kThreads = 256;

constexpr int kPixelShuffleHH = 16;
constexpr int kPixelShuffleWW = 16;
constexpr int kPixelShuffleSpatial = kPixelShuffleHH * kPixelShuffleWW;  // 256
constexpr int kInputChannels = 3;
constexpr int kProjChannels = kInputChannels * kPixelShuffleSpatial;  // 768

// ---------------------------------------------------------------------------
// Type tag for our manual dispatch. Same pattern as
// ``color_corrector_adain_cuda.cu``: bypass ``AT_DISPATCH_FLOATING_TYPES_AND2``
// and feed the kernel templates the CUDA-native element types directly,
// reinterpret-casting ``at::Half`` / ``at::BFloat16`` -> ``__half`` /
// ``__nv_bfloat16``.
// ---------------------------------------------------------------------------

template <typename T>
struct CudaTypeTag {
    using type = T;
};

// PyTorch's cpp_extension build sets ``__CUDA_NO_HALF_CONVERSIONS__`` and
// ``__CUDA_NO_BFLOAT16_CONVERSIONS__``; call the explicit intrinsics.

__device__ inline float to_float_scalar(float x) {
    return x;
}
__device__ inline float to_float_scalar(__half x) {
    return __half2float(x);
}
__device__ inline float to_float_scalar(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ inline T from_float_scalar(float x);
template <>
__device__ inline float from_float_scalar<float>(float x) {
    return x;
}
template <>
__device__ inline __half from_float_scalar<__half>(float x) {
    return __float2half_rn(x);
}
template <>
__device__ inline __nv_bfloat16 from_float_scalar<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

// ---------------------------------------------------------------------------
// Cubic-convolution helpers (mirror ATen ``UpSample.cuh``).
// ---------------------------------------------------------------------------

__device__ inline float cubic_convolution1(float x, float A) {
    return ((A + 2.0f) * x - (A + 3.0f)) * x * x + 1.0f;
}

__device__ inline float cubic_convolution2(float x, float A) {
    return ((A * x - 5.0f * A) * x + 8.0f * A) * x - 4.0f * A;
}

__device__ inline void get_cubic_upsample_coefficients(
    float coeffs[4], float t) {
    constexpr float A = -0.75f;
    const float x1 = t;
    coeffs[0] = cubic_convolution2(x1 + 1.0f, A);
    coeffs[1] = cubic_convolution1(x1, A);
    const float x2 = 1.0f - t;
    coeffs[2] = cubic_convolution1(x2, A);
    coeffs[3] = cubic_convolution2(x2 + 1.0f, A);
}

__device__ inline float cubic_interp1d(
    float x0, float x1, float x2, float x3, float t) {
    float coeffs[4];
    get_cubic_upsample_coefficients(coeffs, t);
    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

__device__ inline int clamp_index(int v, int lo, int hi) {
    return max(min(v, hi), lo);
}

// ---------------------------------------------------------------------------
// Source-coordinate mapping for ``align_corners=False``. Mirrors ATen's
// ``compute_source_index`` (cubic_align_corners=false branch):
//   src = (dst + 0.5) * scale - 0.5
// where ``scale = src_size / dst_size``.
// ---------------------------------------------------------------------------

__device__ inline float compute_src_coord(int dst, float scale) {
    return (static_cast<float>(dst) + 0.5f) * scale - 0.5f;
}

// ---------------------------------------------------------------------------
// One thread per ``proj_input`` element. Output coords are decoded from
// the linear thread id; bicubic + replicate-pad-left are folded in via
// ``t_in = max(0, t_padded - n_left_padding)`` and a per-output bicubic
// 4x4 weighted sum.
//
// ``last_upres`` is side-written when ``t_padded >= n_left_padding`` --
// the unique mapping between (c_ps, h_ps, w_ps) and (c, y_full, x_full)
// makes each ``last_upres`` element the responsibility of exactly one
// thread (no atomics).
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void bicubic_pixelshuffle_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ proj_input,
    scalar_t* __restrict__ last_upres,
    int B,
    int T_raw,
    int H,
    int W,
    int target_T,
    int target_H,
    int target_W,
    int H_out,
    int W_out,
    int n_left_padding,
    float h_scale,
    float w_scale) {
    const int64_t total = static_cast<int64_t>(B) * kProjChannels * target_T *
                          H_out * W_out;
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    int64_t tmp = idx;
    const int w_ps = static_cast<int>(tmp % W_out);
    tmp /= W_out;
    const int h_ps = static_cast<int>(tmp % H_out);
    tmp /= H_out;
    const int f = static_cast<int>(tmp % target_T);
    tmp /= target_T;
    const int c_ps = static_cast<int>(tmp % kProjChannels);
    const int b = static_cast<int>(tmp / kProjChannels);

    const int ww_idx = c_ps % kPixelShuffleWW;
    const int hh_idx = (c_ps / kPixelShuffleWW) % kPixelShuffleHH;
    const int c = c_ps / kPixelShuffleSpatial;

    const int t_padded = f;
    const int y_full = h_ps * kPixelShuffleHH + hh_idx;
    const int x_full = w_ps * kPixelShuffleWW + ww_idx;

    const int t_in = (t_padded >= n_left_padding)
                         ? (t_padded - n_left_padding)
                         : 0;

    const float y_src = compute_src_coord(y_full, h_scale);
    const float x_src = compute_src_coord(x_full, w_scale);

    const int y_floor = static_cast<int>(floorf(y_src));
    const int x_floor = static_cast<int>(floorf(x_src));
    const float ty = y_src - static_cast<float>(y_floor);
    const float tx = x_src - static_cast<float>(x_floor);

    // Frame base pointer: input[b, c, t_in, :, :].
    const int64_t frame_off =
        ((((int64_t)b * kInputChannels) + c) * T_raw + t_in) * H * W;
    const scalar_t* frame = input + frame_off;

    float row_results[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int yy = clamp_index(y_floor - 1 + i, 0, H - 1);
        const scalar_t* row = frame + static_cast<int64_t>(yy) * W;
        const int x0 = clamp_index(x_floor - 1, 0, W - 1);
        const int x1 = clamp_index(x_floor + 0, 0, W - 1);
        const int x2 = clamp_index(x_floor + 1, 0, W - 1);
        const int x3 = clamp_index(x_floor + 2, 0, W - 1);
        row_results[i] = cubic_interp1d(
            to_float_scalar(row[x0]),
            to_float_scalar(row[x1]),
            to_float_scalar(row[x2]),
            to_float_scalar(row[x3]),
            tx);
    }
    const float result = cubic_interp1d(
        row_results[0],
        row_results[1],
        row_results[2],
        row_results[3],
        ty);

    const scalar_t result_t = from_float_scalar<scalar_t>(result);

    const int64_t proj_off =
        ((((int64_t)b * kProjChannels) + c_ps) * target_T + f) * H_out * W_out +
        static_cast<int64_t>(h_ps) * W_out + w_ps;
    proj_input[proj_off] = result_t;

    if (t_padded >= n_left_padding) {
        const int64_t lu_off =
            ((((int64_t)b * kInputChannels) + c) * T_raw + t_in) * target_H *
                target_W +
            static_cast<int64_t>(y_full) * target_W + x_full;
        last_upres[lu_off] = result_t;
    }
}

// ---------------------------------------------------------------------------
// Host wrapper. Returns ``(proj_input, last_upres)``.
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> bicubic_pixelshuffle_forward_cuda(
    torch::Tensor input,
    int64_t target_T_arg,
    int64_t target_H_arg,
    int64_t target_W_arg,
    int64_t n_left_padding_arg) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "input must be (B, 3, T_raw, H, W)");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(
        input.size(1) == kInputChannels,
        "input channel dim must be ",
        kInputChannels,
        ", got ",
        input.size(1));

    const int B = static_cast<int>(input.size(0));
    const int T_raw = static_cast<int>(input.size(2));
    const int H = static_cast<int>(input.size(3));
    const int W = static_cast<int>(input.size(4));
    const int target_T = static_cast<int>(target_T_arg);
    const int target_H = static_cast<int>(target_H_arg);
    const int target_W = static_cast<int>(target_W_arg);
    const int n_left_padding = static_cast<int>(n_left_padding_arg);

    TORCH_CHECK(B > 0 && T_raw > 0 && H > 0 && W > 0, "input must be non-empty");
    TORCH_CHECK(
        target_T == T_raw + n_left_padding,
        "target_T (",
        target_T,
        ") must equal T_raw (",
        T_raw,
        ") + n_left_padding (",
        n_left_padding,
        ")");
    TORCH_CHECK(
        n_left_padding >= 0,
        "n_left_padding must be non-negative, got ",
        n_left_padding);
    TORCH_CHECK(
        target_H % kPixelShuffleHH == 0 && target_W % kPixelShuffleWW == 0,
        "target_H (",
        target_H,
        ") and target_W (",
        target_W,
        ") must be divisible by ",
        kPixelShuffleHH,
        " and ",
        kPixelShuffleWW);

    const int H_out = target_H / kPixelShuffleHH;
    const int W_out = target_W / kPixelShuffleWW;

    auto proj_input = torch::empty(
        {B, kProjChannels, target_T, H_out, W_out}, input.options());
    auto last_upres = torch::empty(
        {B, kInputChannels, T_raw, target_H, target_W}, input.options());

    const float h_scale =
        static_cast<float>(H) / static_cast<float>(target_H);
    const float w_scale =
        static_cast<float>(W) / static_cast<float>(target_W);

    const int64_t total = static_cast<int64_t>(B) * kProjChannels * target_T *
                          H_out * W_out;
    const int64_t blocks =
        (total + static_cast<int64_t>(kThreads) - 1) /
        static_cast<int64_t>(kThreads);
    TORCH_CHECK(
        blocks <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "grid size overflows int (too many output elements)");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto run_dispatch = [&](auto type_tag) {
        using cuda_t = typename decltype(type_tag)::type;
        cuda_t* in_ptr = reinterpret_cast<cuda_t*>(input.data_ptr());
        cuda_t* proj_ptr = reinterpret_cast<cuda_t*>(proj_input.data_ptr());
        cuda_t* lu_ptr = reinterpret_cast<cuda_t*>(last_upres.data_ptr());

        bicubic_pixelshuffle_kernel<cuda_t>
            <<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
                in_ptr,
                proj_ptr,
                lu_ptr,
                B,
                T_raw,
                H,
                W,
                target_T,
                target_H,
                target_W,
                H_out,
                W_out,
                n_left_padding,
                h_scale,
                w_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };

    switch (input.scalar_type()) {
        case at::ScalarType::Float:
            run_dispatch(CudaTypeTag<float>{});
            break;
        case at::ScalarType::Half:
            run_dispatch(CudaTypeTag<__half>{});
            break;
        case at::ScalarType::BFloat16:
            run_dispatch(CudaTypeTag<__nv_bfloat16>{});
            break;
        default:
            TORCH_CHECK(
                false,
                "bicubic_pixelshuffle_cuda only supports float32, float16, "
                "and bfloat16, got: ",
                input.scalar_type());
    }

    return std::make_tuple(proj_input, last_upres);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "bicubic_pixelshuffle_forward",
        &bicubic_pixelshuffle_forward_cuda,
        "Fused bicubic upres + temporal replicate-pad-left + 16x16 spatial "
        "pixel-shuffle for the FlashVSR encoder. Returns (proj_input, "
        "last_upres).");
}
