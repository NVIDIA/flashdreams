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
#include "../common/framework.h"

//------------------------------------------------------------------------
// Input check helpers.
//------------------------------------------------------------------------

#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

#define NVDR_CHECK_DEVICE(...) do { TORCH_CHECK(at::cuda::check_device({__VA_ARGS__}), __func__, "(): Inputs " #__VA_ARGS__ " must reside on the same GPU device") } while(0)
#define NVDR_CHECK_CPU(...) do { nvdr_check_cpu({__VA_ARGS__}, __func__, "(): Inputs " #__VA_ARGS__ " must reside on CPU"); } while(0)
#define NVDR_CHECK_CONTIGUOUS(...) do { nvdr_check_contiguous({__VA_ARGS__}, __func__, "(): Inputs " #__VA_ARGS__ " must be contiguous tensors"); } while(0)
#define NVDR_CHECK_F32(...) do { nvdr_check_f32({__VA_ARGS__}, __func__, "(): Inputs " #__VA_ARGS__ " must be float32 tensors"); } while(0)
#define NVDR_CHECK_I32(...) do { nvdr_check_i32({__VA_ARGS__}, __func__, "(): Inputs " #__VA_ARGS__ " must be int32 tensors"); } while(0)
inline void nvdr_check_cpu(at::ArrayRef<at::Tensor> ts,        const char* func, const char* err_msg) { for (const at::Tensor& t : ts) TORCH_CHECK(t.device().type() == c10::DeviceType::CPU, func, err_msg); }
inline void nvdr_check_contiguous(at::ArrayRef<at::Tensor> ts, const char* func, const char* err_msg) { for (const at::Tensor& t : ts) TORCH_CHECK(t.is_contiguous(), func, err_msg); }
inline void nvdr_check_f32(at::ArrayRef<at::Tensor> ts,        const char* func, const char* err_msg) { for (const at::Tensor& t : ts) TORCH_CHECK(t.dtype() == torch::kFloat32, func, err_msg); }
inline void nvdr_check_i32(at::ArrayRef<at::Tensor> ts,        const char* func, const char* err_msg) { for (const at::Tensor& t : ts) TORCH_CHECK(t.dtype() == torch::kInt32, func, err_msg); }
//------------------------------------------------------------------------
