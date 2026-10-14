/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

#include <torch/extension.h>

namespace omnidreams_singleview {

pybind11::dict native_tensor_descriptor(const torch::Tensor& tensor);
pybind11::dict native_tensor_ref_descriptor(const torch::Tensor& tensor);
pybind11::dict workspace_allocation_plan(
    torch::Tensor workspace,
    const std::vector<int64_t>& byte_sizes,
    int64_t alignment);
torch::Tensor prepare_contiguous(const torch::Tensor& input);
torch::Tensor zero_workspace_(torch::Tensor workspace);
void bind_native_primitives(pybind11::module_& module);
torch::Tensor prepare_contiguous_cuda(const torch::Tensor& input);
void zero_workspace_cuda(torch::Tensor workspace);

}  // namespace omnidreams_singleview
