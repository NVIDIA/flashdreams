// AdaIN color correction CUDA extension.
//
// Three execution paths, all driven from `adain_forward_5d_cuda`:
//
//   1. cooperative single-kernel fused fast path (preferred when the device
//      supports cooperative launches and the inputs are contiguous + aligned).
//      A persistent kernel splits work via a global atomic counter, computes
//      per-row sum/sumsq stats, performs a manual grid-wide barrier, then
//      streams the normalize+clamp output without re-launching.
//
//   2. two-kernel fast path (vectorized, contiguous-only). Same kernels as
//      the fused path, but launched as separate stats and normalize kernels.
//      Used when cooperative launch is unavailable or the chosen grid does
//      not fit in the SM's resident-block budget.
//
//   3. two-kernel strided fallback. Per-pixel index reconstruction with
//      arbitrary strides. Used when the inputs are not standard-contiguous.
//
// Stage-1 wins applied across all paths:
//   - normalize math pre-folded into (scale, bias) and a single FMA + clamp;
//   - warp-shuffle reduction in the stats phase, only one __syncthreads
//     across the 8 per-block warps;
//   - vectorized loads/stores via __nv_bfloat162 / __half2 / float4;
//   - tile size chosen per-shape from {1024, 2048, 4096, 8192}.
//
// Stage-2: the host wrapper sets a stream-level L2 access policy window
// hinting the content tensor as persisting (so the second pass hits L2),
// and the normalize kernel uses streaming stores so the output does not
// pollute L2.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;  // 8

// ---------------------------------------------------------------------------
// Type tag for our manual dispatch. We bypass AT_DISPATCH_FLOATING_TYPES_AND2
// because that macro also dispatches `double`, which we don't support, and
// because we want to feed the kernel templates the CUDA-native element types
// (`__half` / `__nv_bfloat16`) directly rather than at::Half / at::BFloat16.
// at::Half / at::BFloat16 share the layout of __half / __nv_bfloat16 so a
// reinterpret_cast on raw pointers is well-formed.
// ---------------------------------------------------------------------------

template <typename T>
struct CudaTypeTag {
    using type = T;
};

// ---------------------------------------------------------------------------
// Scalar conversion helpers. PyTorch's cpp_extension build sets
// __CUDA_NO_HALF_CONVERSIONS__ and __CUDA_NO_BFLOAT16_CONVERSIONS__ which
// disable the implicit/explicit conversions between half-precision types and
// float, so we must call the explicit intrinsics.
// ---------------------------------------------------------------------------

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
// Vector traits + load/store helpers
// ---------------------------------------------------------------------------

template <typename scalar_t>
struct VecTraits;

template <>
struct VecTraits<float> {
    using vec_t = float4;
    static constexpr int width = 4;
};

template <>
struct VecTraits<__half> {
    using vec_t = __half2;
    static constexpr int width = 2;
};

template <>
struct VecTraits<__nv_bfloat16> {
    using vec_t = __nv_bfloat162;
    static constexpr int width = 2;
};

__device__ inline void vec_to_floats(float4 v, float (&out)[4]) {
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
    out[3] = v.w;
}
__device__ inline void vec_to_floats(__half2 v, float (&out)[2]) {
    float2 f = __half22float2(v);
    out[0] = f.x;
    out[1] = f.y;
}
__device__ inline void vec_to_floats(__nv_bfloat162 v, float (&out)[2]) {
    float2 f = __bfloat1622float2(v);
    out[0] = f.x;
    out[1] = f.y;
}

template <typename scalar_t>
__device__ inline typename VecTraits<scalar_t>::vec_t floats_to_vec(
    const float (&in)[VecTraits<scalar_t>::width]);

template <>
__device__ inline float4 floats_to_vec<float>(const float (&in)[4]) {
    return make_float4(in[0], in[1], in[2], in[3]);
}
template <>
__device__ inline __half2 floats_to_vec<__half>(const float (&in)[2]) {
    return __floats2half2_rn(in[0], in[1]);
}
template <>
__device__ inline __nv_bfloat162 floats_to_vec<__nv_bfloat16>(const float (&in)[2]) {
    return __floats2bfloat162_rn(in[0], in[1]);
}

// Streaming store wrappers: __stwt natively supports float / float4 but not
// the half-precision packed types, so we reinterpret as the matching
// integer type via a union (well-defined for type punning under nvcc).
__device__ inline void streaming_store(float* ptr, float val) {
    __stwt(ptr, val);
}
__device__ inline void streaming_store(float4* ptr, float4 val) {
    __stwt(ptr, val);
}
__device__ inline void streaming_store(__half* ptr, __half val) {
    union {
        __half h;
        unsigned short u;
    } cvt = {val};
    __stwt(reinterpret_cast<unsigned short*>(ptr), cvt.u);
}
__device__ inline void streaming_store(__nv_bfloat16* ptr, __nv_bfloat16 val) {
    union {
        __nv_bfloat16 h;
        unsigned short u;
    } cvt = {val};
    __stwt(reinterpret_cast<unsigned short*>(ptr), cvt.u);
}
__device__ inline void streaming_store(__half2* ptr, __half2 val) {
    union {
        __half2 h;
        unsigned int u;
    } cvt = {val};
    __stwt(reinterpret_cast<unsigned int*>(ptr), cvt.u);
}
__device__ inline void streaming_store(__nv_bfloat162* ptr, __nv_bfloat162 val) {
    union {
        __nv_bfloat162 h;
        unsigned int u;
    } cvt = {val};
    __stwt(reinterpret_cast<unsigned int*>(ptr), cvt.u);
}

// ---------------------------------------------------------------------------
// Block reduction over four scalars (sum_c, sumsq_c, sum_s, sumsq_s).
// Final result lives in thread 0 only.
// ---------------------------------------------------------------------------

__device__ inline void block_reduce_4(
    float& a,
    float& b,
    float& c,
    float& d,
    float* sm /* size: kWarpsPerBlock * 4 */) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, off);
        b += __shfl_xor_sync(0xffffffffu, b, off);
        c += __shfl_xor_sync(0xffffffffu, c, off);
        d += __shfl_xor_sync(0xffffffffu, d, off);
    }
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        sm[warp * 4 + 0] = a;
        sm[warp * 4 + 1] = b;
        sm[warp * 4 + 2] = c;
        sm[warp * 4 + 3] = d;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ta = 0.0f, tb = 0.0f, tc = 0.0f, td = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; w++) {
            ta += sm[w * 4 + 0];
            tb += sm[w * 4 + 1];
            tc += sm[w * 4 + 2];
            td += sm[w * 4 + 3];
        }
        a = ta;
        b = tb;
        c = tc;
        d = td;
    }
}

// ---------------------------------------------------------------------------
// Manual grid-wide barrier. Requires that every block in the launch is
// concurrently resident (which is guaranteed by `cudaLaunchCooperativeKernel`
// with a grid sized at most num_sms * blocks_per_sm).
// ---------------------------------------------------------------------------

__device__ inline void manual_grid_sync(
    unsigned int* counter,
    unsigned int num_blocks) {
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        atomicAdd(counter, 1u);
        while (atomicAdd(counter, 0u) < num_blocks) {
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Strided two-kernel path (slow but always correct).
// ---------------------------------------------------------------------------

template <typename scalar_t, int BLOCK_PIXELS>
__global__ void adain_stats_strided_kernel(
    const scalar_t* __restrict__ content,
    const scalar_t* __restrict__ style,
    float* __restrict__ stats,
    int64_t C,
    int64_t T,
    int64_t W,
    int64_t c_sb,
    int64_t c_sc,
    int64_t c_st,
    int64_t c_sh,
    int64_t c_sw,
    int64_t s_sb,
    int64_t s_sc,
    int64_t s_st,
    int64_t s_sh,
    int64_t s_sw,
    int64_t rows,
    int64_t num_pixels,
    int64_t num_tiles) {
    const int64_t row = blockIdx.x;
    const int64_t tile = blockIdx.y;
    if (row >= rows || tile >= num_tiles) {
        return;
    }
    const int64_t b = row / (C * T);
    const int64_t rem = row - b * C * T;
    const int64_t cc = rem / T;
    const int64_t tt = rem - cc * T;
    const int64_t tile_start = tile * BLOCK_PIXELS;
    const int64_t tile_end_unclamped = tile_start + BLOCK_PIXELS;
    const int64_t tile_end =
        (tile_end_unclamped < num_pixels) ? tile_end_unclamped : num_pixels;

    float cs = 0.0f, css = 0.0f, ss = 0.0f, sss = 0.0f;
    for (int64_t pixel = tile_start + threadIdx.x; pixel < tile_end;
         pixel += blockDim.x) {
        const int64_t hh = pixel / W;
        const int64_t ww = pixel - hh * W;
        const int64_t coff =
            b * c_sb + cc * c_sc + tt * c_st + hh * c_sh + ww * c_sw;
        const int64_t soff =
            b * s_sb + cc * s_sc + tt * s_st + hh * s_sh + ww * s_sw;
        const float cv = to_float_scalar(content[coff]);
        const float sv = to_float_scalar(style[soff]);
        cs += cv;
        css += cv * cv;
        ss += sv;
        sss += sv * sv;
    }

    __shared__ float sm[kWarpsPerBlock * 4];
    block_reduce_4(cs, css, ss, sss, sm);

    if (threadIdx.x == 0) {
        float* row_stats = stats + row * 4;
        atomicAdd(row_stats + 0, cs);
        atomicAdd(row_stats + 1, css);
        atomicAdd(row_stats + 2, ss);
        atomicAdd(row_stats + 3, sss);
    }
}

template <typename scalar_t, int BLOCK_PIXELS>
__global__ void adain_normalize_strided_kernel(
    const scalar_t* __restrict__ content,
    scalar_t* __restrict__ out,
    const float* __restrict__ stats,
    int64_t C,
    int64_t T,
    int64_t H,
    int64_t W,
    int64_t c_sb,
    int64_t c_sc,
    int64_t c_st,
    int64_t c_sh,
    int64_t c_sw,
    int64_t rows,
    int64_t num_pixels,
    int64_t num_tiles,
    float clip_min,
    float clip_max,
    float eps) {
    const int64_t row = blockIdx.x;
    const int64_t tile = blockIdx.y;
    if (row >= rows || tile >= num_tiles) {
        return;
    }
    const float* row_stats = stats + row * 4;
    const float inv_count = 1.0f / static_cast<float>(num_pixels);
    const float content_mean = row_stats[0] * inv_count;
    const float style_mean = row_stats[2] * inv_count;
    const float content_var =
        fmaxf(row_stats[1] * inv_count - content_mean * content_mean, 0.0f);
    const float style_var =
        fmaxf(row_stats[3] * inv_count - style_mean * style_mean, 0.0f);
    const float style_std = sqrtf(style_var + eps);
    const float scale = style_std * rsqrtf(content_var + eps);
    const float bias = style_mean - content_mean * scale;

    const int64_t b = row / (C * T);
    const int64_t rem = row - b * C * T;
    const int64_t cc = rem / T;
    const int64_t tt = rem - cc * T;
    const int64_t tile_start = tile * BLOCK_PIXELS;
    const int64_t tile_end_unclamped = tile_start + BLOCK_PIXELS;
    const int64_t tile_end =
        (tile_end_unclamped < num_pixels) ? tile_end_unclamped : num_pixels;

    for (int64_t pixel = tile_start + threadIdx.x; pixel < tile_end;
         pixel += blockDim.x) {
        const int64_t hh = pixel / W;
        const int64_t ww = pixel - hh * W;
        const int64_t coff =
            b * c_sb + cc * c_sc + tt * c_st + hh * c_sh + ww * c_sw;
        const int64_t ooff = (((b * C + cc) * T + tt) * H + hh) * W + ww;
        const float cv = to_float_scalar(content[coff]);
        float corrected = fmaf(cv, scale, bias);
        corrected = fminf(fmaxf(corrected, clip_min), clip_max);
        streaming_store(out + ooff, from_float_scalar<scalar_t>(corrected));
    }
}

// ---------------------------------------------------------------------------
// Vectorized contiguous fast-path kernels.
// ---------------------------------------------------------------------------

template <typename scalar_t, int BLOCK_PIXELS>
__global__ void adain_stats_fast_kernel(
    const scalar_t* __restrict__ content,
    const scalar_t* __restrict__ style,
    float* __restrict__ stats,
    int64_t rows,
    int64_t num_pixels,
    int64_t num_tiles) {
    using vec_t = typename VecTraits<scalar_t>::vec_t;
    constexpr int VW = VecTraits<scalar_t>::width;

    const int64_t row = blockIdx.x;
    const int64_t tile = blockIdx.y;
    if (row >= rows || tile >= num_tiles) {
        return;
    }
    const int64_t row_off = row * num_pixels;
    const int64_t tile_start = tile * BLOCK_PIXELS;
    const int64_t tile_end_unclamped = tile_start + BLOCK_PIXELS;
    const int64_t tile_end =
        (tile_end_unclamped < num_pixels) ? tile_end_unclamped : num_pixels;

    const vec_t* content_v = reinterpret_cast<const vec_t*>(content);
    const vec_t* style_v = reinterpret_cast<const vec_t*>(style);
    const int64_t row_voff = row_off / VW;
    const int64_t v_start = tile_start / VW;
    const int64_t v_end = tile_end / VW;

    float cs = 0.0f, css = 0.0f, ss = 0.0f, sss = 0.0f;
    for (int64_t v = v_start + threadIdx.x; v < v_end; v += blockDim.x) {
        vec_t cv = content_v[row_voff + v];
        vec_t sv = style_v[row_voff + v];
        float cf[VW];
        float sf[VW];
        vec_to_floats(cv, cf);
        vec_to_floats(sv, sf);
#pragma unroll
        for (int i = 0; i < VW; i++) {
            cs += cf[i];
            css += cf[i] * cf[i];
            ss += sf[i];
            sss += sf[i] * sf[i];
        }
    }

    __shared__ float sm[kWarpsPerBlock * 4];
    block_reduce_4(cs, css, ss, sss, sm);

    if (threadIdx.x == 0) {
        float* row_stats = stats + row * 4;
        atomicAdd(row_stats + 0, cs);
        atomicAdd(row_stats + 1, css);
        atomicAdd(row_stats + 2, ss);
        atomicAdd(row_stats + 3, sss);
    }
}

template <typename scalar_t, int BLOCK_PIXELS>
__global__ void adain_normalize_fast_kernel(
    const scalar_t* __restrict__ content,
    scalar_t* __restrict__ out,
    const float* __restrict__ stats,
    int64_t rows,
    int64_t num_pixels,
    int64_t num_tiles,
    float clip_min,
    float clip_max,
    float eps) {
    using vec_t = typename VecTraits<scalar_t>::vec_t;
    constexpr int VW = VecTraits<scalar_t>::width;

    const int64_t row = blockIdx.x;
    const int64_t tile = blockIdx.y;
    if (row >= rows || tile >= num_tiles) {
        return;
    }
    const float* row_stats = stats + row * 4;
    const float inv_count = 1.0f / static_cast<float>(num_pixels);
    const float content_mean = row_stats[0] * inv_count;
    const float style_mean = row_stats[2] * inv_count;
    const float content_var =
        fmaxf(row_stats[1] * inv_count - content_mean * content_mean, 0.0f);
    const float style_var =
        fmaxf(row_stats[3] * inv_count - style_mean * style_mean, 0.0f);
    const float style_std = sqrtf(style_var + eps);
    const float scale = style_std * rsqrtf(content_var + eps);
    const float bias = style_mean - content_mean * scale;

    const int64_t row_off = row * num_pixels;
    const int64_t tile_start = tile * BLOCK_PIXELS;
    const int64_t tile_end_unclamped = tile_start + BLOCK_PIXELS;
    const int64_t tile_end =
        (tile_end_unclamped < num_pixels) ? tile_end_unclamped : num_pixels;

    const vec_t* content_v = reinterpret_cast<const vec_t*>(content);
    vec_t* out_v = reinterpret_cast<vec_t*>(out);
    const int64_t row_voff = row_off / VW;
    const int64_t v_start = tile_start / VW;
    const int64_t v_end = tile_end / VW;

    for (int64_t v = v_start + threadIdx.x; v < v_end; v += blockDim.x) {
        vec_t cv = content_v[row_voff + v];
        float cf[VW];
        vec_to_floats(cv, cf);
        float of[VW];
#pragma unroll
        for (int i = 0; i < VW; i++) {
            float corrected = fmaf(cf[i], scale, bias);
            corrected = fminf(fmaxf(corrected, clip_min), clip_max);
            of[i] = corrected;
        }
        vec_t ov = floats_to_vec<scalar_t>(of);
        streaming_store(out_v + row_voff + v, ov);
    }
}

// ---------------------------------------------------------------------------
// Cooperative single-kernel fused fast path.
//
// Layout of `counters` (3 unsigned ints, zeroed before launch):
//   counters[0]: phase 1 work-stealing index
//   counters[1]: phase 2 work-stealing index
//   counters[2]: grid-sync barrier counter
// ---------------------------------------------------------------------------

template <typename scalar_t, int BLOCK_PIXELS>
__global__ void adain_fused_fast_kernel(
    const scalar_t* __restrict__ content,
    const scalar_t* __restrict__ style,
    scalar_t* __restrict__ out,
    float* __restrict__ stats,
    unsigned int* __restrict__ counters,
    int64_t rows,
    int64_t num_pixels,
    int64_t num_tiles,
    int64_t total_tiles,
    float clip_min,
    float clip_max,
    float eps,
    unsigned int num_blocks) {
    using vec_t = typename VecTraits<scalar_t>::vec_t;
    constexpr int VW = VecTraits<scalar_t>::width;

    __shared__ unsigned int shared_idx;
    __shared__ float sm[kWarpsPerBlock * 4];

    const float inv_count = 1.0f / static_cast<float>(num_pixels);

    // Phase 1: per-tile stats reduction.
    while (true) {
        __syncthreads();
        if (threadIdx.x == 0) {
            shared_idx = atomicAdd(counters + 0, 1u);
        }
        __syncthreads();
        const unsigned int idx = shared_idx;
        if (static_cast<int64_t>(idx) >= total_tiles) {
            break;
        }
        const int64_t row = static_cast<int64_t>(idx) / num_tiles;
        const int64_t tile = static_cast<int64_t>(idx) - row * num_tiles;
        const int64_t row_off = row * num_pixels;
        const int64_t tile_start = tile * BLOCK_PIXELS;
        const int64_t tile_end_unclamped = tile_start + BLOCK_PIXELS;
        const int64_t tile_end =
            (tile_end_unclamped < num_pixels) ? tile_end_unclamped : num_pixels;

        const vec_t* content_v = reinterpret_cast<const vec_t*>(content);
        const vec_t* style_v = reinterpret_cast<const vec_t*>(style);
        const int64_t row_voff = row_off / VW;
        const int64_t v_start = tile_start / VW;
        const int64_t v_end = tile_end / VW;

        float cs = 0.0f, css = 0.0f, ss = 0.0f, sss = 0.0f;
        for (int64_t v = v_start + threadIdx.x; v < v_end; v += blockDim.x) {
            vec_t cv = content_v[row_voff + v];
            vec_t sv = style_v[row_voff + v];
            float cf[VW];
            float sf[VW];
            vec_to_floats(cv, cf);
            vec_to_floats(sv, sf);
#pragma unroll
            for (int i = 0; i < VW; i++) {
                cs += cf[i];
                css += cf[i] * cf[i];
                ss += sf[i];
                sss += sf[i] * sf[i];
            }
        }
        block_reduce_4(cs, css, ss, sss, sm);
        if (threadIdx.x == 0) {
            float* row_stats = stats + row * 4;
            atomicAdd(row_stats + 0, cs);
            atomicAdd(row_stats + 1, css);
            atomicAdd(row_stats + 2, ss);
            atomicAdd(row_stats + 3, sss);
        }
    }

    // Grid-wide barrier so phase 2 sees finalized stats.
    manual_grid_sync(counters + 2, num_blocks);

    // Phase 2: normalize + clamp + streaming store.
    while (true) {
        __syncthreads();
        if (threadIdx.x == 0) {
            shared_idx = atomicAdd(counters + 1, 1u);
        }
        __syncthreads();
        const unsigned int idx = shared_idx;
        if (static_cast<int64_t>(idx) >= total_tiles) {
            break;
        }
        const int64_t row = static_cast<int64_t>(idx) / num_tiles;
        const int64_t tile = static_cast<int64_t>(idx) - row * num_tiles;

        const float* row_stats = stats + row * 4;
        const float content_mean = row_stats[0] * inv_count;
        const float style_mean = row_stats[2] * inv_count;
        const float content_var = fmaxf(
            row_stats[1] * inv_count - content_mean * content_mean, 0.0f);
        const float style_var = fmaxf(
            row_stats[3] * inv_count - style_mean * style_mean, 0.0f);
        const float style_std = sqrtf(style_var + eps);
        const float scale = style_std * rsqrtf(content_var + eps);
        const float bias = style_mean - content_mean * scale;

        const int64_t row_off = row * num_pixels;
        const int64_t tile_start = tile * BLOCK_PIXELS;
        const int64_t tile_end_unclamped = tile_start + BLOCK_PIXELS;
        const int64_t tile_end =
            (tile_end_unclamped < num_pixels) ? tile_end_unclamped : num_pixels;

        const vec_t* content_v = reinterpret_cast<const vec_t*>(content);
        vec_t* out_v = reinterpret_cast<vec_t*>(out);
        const int64_t row_voff = row_off / VW;
        const int64_t v_start = tile_start / VW;
        const int64_t v_end = tile_end / VW;

        for (int64_t v = v_start + threadIdx.x; v < v_end; v += blockDim.x) {
            vec_t cv = content_v[row_voff + v];
            float cf[VW];
            vec_to_floats(cv, cf);
            float of[VW];
#pragma unroll
            for (int i = 0; i < VW; i++) {
                float corrected = fmaf(cf[i], scale, bias);
                corrected = fminf(fmaxf(corrected, clip_min), clip_max);
                of[i] = corrected;
            }
            vec_t ov = floats_to_vec<scalar_t>(of);
            streaming_store(out_v + row_voff + v, ov);
        }
    }
}

// ---------------------------------------------------------------------------
// Device caps + per-shape tile size selection.
// ---------------------------------------------------------------------------

struct DeviceCaps {
    bool cooperative_launch = false;
    bool persisting_l2 = false;
    int num_sms = 0;
    size_t access_policy_max_window_size = 0;
    size_t persisting_l2_max_size = 0;
};

DeviceCaps query_caps(int device) {
    cudaDeviceProp prop;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    DeviceCaps c;
    c.cooperative_launch = prop.cooperativeLaunch != 0;
    c.persisting_l2 = prop.persistingL2CacheMaxSize > 0;
    c.num_sms = prop.multiProcessorCount;
    c.access_policy_max_window_size =
        static_cast<size_t>(prop.accessPolicyMaxWindowSize);
    c.persisting_l2_max_size =
        static_cast<size_t>(prop.persistingL2CacheMaxSize);
    return c;
}

const DeviceCaps& caps_for_device(int device) {
    static std::mutex m;
    static std::unordered_map<int, DeviceCaps> cache;
    std::lock_guard<std::mutex> lk(m);
    auto it = cache.find(device);
    if (it == cache.end()) {
        it = cache.emplace(device, query_caps(device)).first;
    }
    return it->second;
}

int pick_block_pixels(int64_t num_pixels) {
    if (num_pixels >= 65536) {
        return 8192;
    }
    if (num_pixels >= 16384) {
        return 4096;
    }
    if (num_pixels >= 4096) {
        return 2048;
    }
    return 1024;
}

template <typename cuda_t>
bool fast_eligible_cuda(const torch::Tensor& t, int64_t num_pixels) {
    using VT = typename VecTraits<cuda_t>::vec_t;
    constexpr int VW = VecTraits<cuda_t>::width;
    if (!t.is_contiguous()) {
        return false;
    }
    if (num_pixels % VW != 0) {
        return false;
    }
    auto p = reinterpret_cast<uintptr_t>(t.data_ptr());
    if (p % alignof(VT) != 0) {
        return false;
    }
    return true;
}

// Compile-time dispatch over the supported BLOCK_PIXELS values.
template <typename Func>
void with_block_pixels(int bp, Func&& f) {
    switch (bp) {
        case 1024:
            f(std::integral_constant<int, 1024>{});
            break;
        case 2048:
            f(std::integral_constant<int, 2048>{});
            break;
        case 4096:
            f(std::integral_constant<int, 4096>{});
            break;
        case 8192:
            f(std::integral_constant<int, 8192>{});
            break;
        default:
            TORCH_CHECK(false, "Unsupported block_pixels: ", bp);
    }
}

// ---------------------------------------------------------------------------
// Host wrapper.
// ---------------------------------------------------------------------------

torch::Tensor adain_forward_5d_cuda(
    torch::Tensor content,
    torch::Tensor style,
    double clip_min,
    double clip_max,
    double eps) {
    TORCH_CHECK(content.is_cuda(), "content must be a CUDA tensor");
    TORCH_CHECK(style.is_cuda(), "style must be a CUDA tensor");
    TORCH_CHECK(
        content.scalar_type() == style.scalar_type(),
        "content and style dtype mismatch");
    TORCH_CHECK(content.sizes() == style.sizes(), "shape mismatch");
    TORCH_CHECK(content.dim() == 5, "input must be (B, C, T, H, W)");
    TORCH_CHECK(content.device() == style.device(), "device mismatch");

    const int64_t B = content.size(0);
    const int64_t C = content.size(1);
    const int64_t T = content.size(2);
    const int64_t H = content.size(3);
    const int64_t W = content.size(4);
    const int64_t rows = B * C * T;
    const int64_t num_pixels = H * W;
    TORCH_CHECK(rows > 0 && num_pixels > 0, "input must be non-empty");

    const int block_pixels = pick_block_pixels(num_pixels);
    const int64_t num_tiles =
        (num_pixels + block_pixels - 1) / block_pixels;
    const int64_t total_tiles = rows * num_tiles;
    TORCH_CHECK(
        rows <= std::numeric_limits<unsigned int>::max(),
        "too many rows for CUDA grid");
    TORCH_CHECK(
        num_tiles <= std::numeric_limits<unsigned int>::max(),
        "too many tiles for CUDA grid");
    TORCH_CHECK(
        total_tiles <= std::numeric_limits<int>::max(),
        "total_tiles overflows int");

    auto out = torch::empty(content.sizes(), content.options());
    auto stats = torch::zeros(
        {rows, 4}, content.options().dtype(torch::kFloat32));

    const int device = content.device().index();
    const auto& caps = caps_for_device(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Stage-2: hint the L2 cache to keep `content` persistent so the second
    // pass (whether two-kernel or fused) hits L2 instead of HBM. Cleared
    // before returning.
    bool l2_set = false;
    if (caps.persisting_l2 && caps.access_policy_max_window_size > 0) {
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.base_ptr = content.data_ptr();
        size_t bytes = static_cast<size_t>(content.numel()) *
                       static_cast<size_t>(content.element_size());
        attr.accessPolicyWindow.num_bytes =
            std::min(bytes, caps.access_policy_max_window_size);
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        if (cudaStreamSetAttribute(
                stream,
                cudaStreamAttributeAccessPolicyWindow,
                &attr) == cudaSuccess) {
            l2_set = true;
        } else {
            (void)cudaGetLastError();
        }
    }

    auto run_dispatch = [&](auto type_tag) {
        using cuda_t = typename decltype(type_tag)::type;
        const bool fast = fast_eligible_cuda<cuda_t>(content, num_pixels) &&
                          fast_eligible_cuda<cuda_t>(style, num_pixels) &&
                          fast_eligible_cuda<cuda_t>(out, num_pixels);

        cuda_t* cptr = reinterpret_cast<cuda_t*>(content.data_ptr());
        cuda_t* sptr = reinterpret_cast<cuda_t*>(style.data_ptr());
        cuda_t* optr = reinterpret_cast<cuda_t*>(out.data_ptr());

            bool launched_coop = false;
            if (fast && caps.cooperative_launch) {
                with_block_pixels(block_pixels, [&](auto bp_tag) {
                    constexpr int BP = decltype(bp_tag)::value;
                    int blocks_per_sm = 0;
                    // Dynamic shared mem is 0 — our kernel uses static
                    // __shared__ declarations only.
                    cudaError_t occ_err =
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &blocks_per_sm,
                            adain_fused_fast_kernel<cuda_t, BP>,
                            kThreads,
                            0);
                    if (occ_err == cudaSuccess && blocks_per_sm > 0) {
                        const int64_t cap_grid = static_cast<int64_t>(
                            caps.num_sms * blocks_per_sm);
                        const int grid_size = static_cast<int>(
                            std::max<int64_t>(
                                1,
                                std::min<int64_t>(cap_grid, total_tiles)));
                        auto counters = torch::zeros(
                            {3},
                            content.options().dtype(torch::kInt32));

                        cuda_t* cptr_arg = cptr;
                        cuda_t* sptr_arg = sptr;
                        cuda_t* optr_arg = optr;
                        auto* stats_ptr = stats.data_ptr<float>();
                        auto* counter_ptr =
                            reinterpret_cast<unsigned int*>(counters.data_ptr());
                        const int64_t rows_arg = rows;
                        const int64_t num_pixels_arg = num_pixels;
                        const int64_t num_tiles_arg = num_tiles;
                        const int64_t total_tiles_arg = total_tiles;
                        const float clip_min_f = static_cast<float>(clip_min);
                        const float clip_max_f = static_cast<float>(clip_max);
                        const float eps_f = static_cast<float>(eps);
                        const unsigned int num_blocks_u =
                            static_cast<unsigned int>(grid_size);

                        void* kargs[] = {
                            (void*)&cptr_arg,
                            (void*)&sptr_arg,
                            (void*)&optr_arg,
                            (void*)&stats_ptr,
                            (void*)&counter_ptr,
                            (void*)&rows_arg,
                            (void*)&num_pixels_arg,
                            (void*)&num_tiles_arg,
                            (void*)&total_tiles_arg,
                            (void*)&clip_min_f,
                            (void*)&clip_max_f,
                            (void*)&eps_f,
                            (void*)&num_blocks_u,
                        };
                        cudaError_t launch_err =
                            cudaLaunchCooperativeKernel(
                                reinterpret_cast<const void*>(
                                    adain_fused_fast_kernel<cuda_t, BP>),
                                dim3(static_cast<unsigned int>(grid_size)),
                                dim3(kThreads),
                                kargs,
                                0,
                                stream);
                        if (launch_err == cudaSuccess) {
                            launched_coop = true;
                            // PyTorch's caching allocator is stream-aware:
                            // the `counters` storage will not be reused by
                            // a later allocation on this stream until this
                            // kernel completes, so RAII free here is safe.
                        } else {
                            (void)cudaGetLastError();
                        }
                    }
                });
            }

            if (!launched_coop && fast) {
                with_block_pixels(block_pixels, [&](auto bp_tag) {
                    constexpr int BP = decltype(bp_tag)::value;
                    dim3 grid(
                        static_cast<unsigned int>(rows),
                        static_cast<unsigned int>(num_tiles));
                    adain_stats_fast_kernel<cuda_t, BP>
                        <<<grid, kThreads, 0, stream>>>(
                            cptr,
                            sptr,
                            stats.data_ptr<float>(),
                            rows,
                            num_pixels,
                            num_tiles);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    adain_normalize_fast_kernel<cuda_t, BP>
                        <<<grid, kThreads, 0, stream>>>(
                            cptr,
                            optr,
                            stats.data_ptr<float>(),
                            rows,
                            num_pixels,
                            num_tiles,
                            static_cast<float>(clip_min),
                            static_cast<float>(clip_max),
                            static_cast<float>(eps));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            } else if (!launched_coop) {
                with_block_pixels(block_pixels, [&](auto bp_tag) {
                    constexpr int BP = decltype(bp_tag)::value;
                    dim3 grid(
                        static_cast<unsigned int>(rows),
                        static_cast<unsigned int>(num_tiles));
                    adain_stats_strided_kernel<cuda_t, BP>
                        <<<grid, kThreads, 0, stream>>>(
                            cptr,
                            sptr,
                            stats.data_ptr<float>(),
                            C,
                            T,
                            W,
                            content.stride(0),
                            content.stride(1),
                            content.stride(2),
                            content.stride(3),
                            content.stride(4),
                            style.stride(0),
                            style.stride(1),
                            style.stride(2),
                            style.stride(3),
                            style.stride(4),
                            rows,
                            num_pixels,
                            num_tiles);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    adain_normalize_strided_kernel<cuda_t, BP>
                        <<<grid, kThreads, 0, stream>>>(
                            cptr,
                            optr,
                            stats.data_ptr<float>(),
                            C,
                            T,
                            H,
                            W,
                            content.stride(0),
                            content.stride(1),
                            content.stride(2),
                            content.stride(3),
                            content.stride(4),
                            rows,
                            num_pixels,
                            num_tiles,
                            static_cast<float>(clip_min),
                            static_cast<float>(clip_max),
                            static_cast<float>(eps));
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            }
    };

    switch (content.scalar_type()) {
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
                "AdaIN CUDA only supports float32, float16, and bfloat16, got: ",
                content.scalar_type());
    }

    if (l2_set) {
        cudaStreamAttrValue clear_attr{};
        cudaStreamSetAttribute(
            stream,
            cudaStreamAttributeAccessPolicyWindow,
            &clear_attr);
    }

    return out;
}

// ---------------------------------------------------------------------------
// Capability query exposed to Python.
// ---------------------------------------------------------------------------

pybind11::dict caps_dict() {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    const auto& c = caps_for_device(device);
    pybind11::dict d;
    d["cooperative_launch"] = c.cooperative_launch;
    d["persisting_l2"] = c.persisting_l2;
    d["num_sms"] = c.num_sms;
    d["access_policy_max_window_size"] = c.access_policy_max_window_size;
    d["persisting_l2_max_size"] = c.persisting_l2_max_size;
    return d;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "adain_forward_5d",
        &adain_forward_5d_cuda,
        "AdaIN color correction over 5D B,C,T,H,W tensors (CUDA, fused)");
    m.def(
        "caps",
        &caps_dict,
        "Device capabilities used by the AdaIN extension");
}
