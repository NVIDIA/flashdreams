// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Framework-specific macros for the Torch plugin.

//------------------------------------------------------------------------
// PyTorch.

#ifdef NVDR_TORCH
#ifndef __CUDACC__
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/numpy.h>
#include <vector>
#endif

// Context parameter passed through the call chain. Not used by any backend
// but kept for source-level compatibility with nvdiffrast-lineage call sites.
#define NVDR_CTX_ARGS void* nvdr_ctx
#define NVDR_CTX_PARAMS nullptr

#define NVDR_CHECK(COND, ERR) do { TORCH_CHECK(COND, ERR) } while(0)

#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { cudaError_t err = CUDA_CALL; TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError(), "[", #CUDA_CALL, ";]"); } while(0)

// ---- Convenience tensor-check helpers (used by CUDA bindings) ----
#ifndef __CUDACC__
namespace nvdr {

inline void check_device(std::initializer_list<torch::Tensor> tensors) {
    for (const auto& t : tensors) {
        TORCH_CHECK(t.device().is_cuda(), "tensor must be CUDA");
    }
}

inline void check_contiguous(std::initializer_list<torch::Tensor> tensors) {
    for (const auto& t : tensors) {
        TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous");
    }
}

inline void check_f32(std::initializer_list<torch::Tensor> tensors) {
    for (const auto& t : tensors) {
        TORCH_CHECK(t.dtype() == torch::kFloat32, "tensor must be float32");
    }
}

inline void check_i32(std::initializer_list<torch::Tensor> tensors) {
    for (const auto& t : tensors) {
        TORCH_CHECK(t.dtype() == torch::kInt32, "tensor must be int32");
    }
}

} // namespace nvdr

#define NVDR_CHECK_DEVICE(...) nvdr::check_device({__VA_ARGS__})
#define NVDR_CHECK_CONTIGUOUS(...) nvdr::check_contiguous({__VA_ARGS__})
#define NVDR_CHECK_F32(...) nvdr::check_f32({__VA_ARGS__})
#define NVDR_CHECK_I32(...) nvdr::check_i32({__VA_ARGS__})

#endif // !__CUDACC__
#endif // NVDR_TORCH

//------------------------------------------------------------------------
