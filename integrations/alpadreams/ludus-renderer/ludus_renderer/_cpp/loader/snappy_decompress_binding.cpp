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

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> batch_snappy_decompress(
    torch::Tensor gpu_buffer,
    torch::Tensor data_offsets,
    torch::Tensor comp_sizes,
    torch::Tensor uncomp_sizes
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_snappy_decompress", &batch_snappy_decompress,
          "Batch Snappy decompress via nvcomp C API (GIL-free)",
          py::call_guard<py::gil_scoped_release>());
}
