// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// Framework-specific macros to enable code sharing between GL and CUDA-only
// plugin variants.

//------------------------------------------------------------------------
// Tensorflow.

#ifdef NVDR_TENSORFLOW
#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/default/logging.h"
using namespace tensorflow;
using namespace tensorflow::shape_inference;
#define NVDR_CTX_ARGS OpKernelContext* _nvdr_ctx
#define NVDR_CTX_PARAMS _nvdr_ctx
#define NVDR_CHECK(COND, ERR) OP_REQUIRES(_nvdr_ctx, COND, errors::Internal(ERR))
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) OP_CHECK_CUDA_ERROR(_nvdr_ctx, CUDA_CALL)
#define NVDR_CHECK_GL_ERROR(GL_CALL) OP_CHECK_GL_ERROR(_nvdr_ctx, GL_CALL)
#endif

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

// Generic condition check (both GL and CUDA-only paths).
#define NVDR_CHECK(COND, ERR) do { TORCH_CHECK(COND, ERR) } while(0)

// CUDA error check -- used by the GL backend and the CUDA rasterizer host code.
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { cudaError_t err = CUDA_CALL; TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError(), "[", #CUDA_CALL, ";]"); } while(0)

// OpenGL error check -- used only by the GL backend (ludus_gl.cpp,
// ludus_timestamped_gl.cpp). Harmlessly defined even when compiling the
// CUDA-only plugin since it is never called there.
#define NVDR_CHECK_GL_ERROR(GL_CALL) do { GL_CALL; GLenum err = glGetError(); TORCH_CHECK(err == GL_NO_ERROR, "OpenGL error: ", getGLErrorString(err), "[", #GL_CALL, ";]"); } while(0)

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
