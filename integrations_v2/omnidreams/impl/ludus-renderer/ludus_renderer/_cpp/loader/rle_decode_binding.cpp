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

torch::Tensor decode_rle_streams(
    torch::Tensor data,
    torch::Tensor page_offset,
    torch::Tensor page_length,
    torch::Tensor page_max_rep,
    torch::Tensor page_max_def,
    torch::Tensor page_num_values,
    torch::Tensor page_out_start,
    int total_values
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_rle_streams", &decode_rle_streams,
          "Decode RLE/bit-packing hybrid streams with SMEM fast path",
          py::call_guard<py::gil_scoped_release>());
}
