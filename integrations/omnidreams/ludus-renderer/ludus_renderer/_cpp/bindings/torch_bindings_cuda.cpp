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

#include "torch_common.inl"
#include "torch_types.h"
#include "../cudaraster/CudaRaster.hpp"
#include <tuple>

//------------------------------------------------------------------------
// Op prototypes.

torch::Tensor ludus_render_fwd_cuda(LudusCudaStateWrapper& stateWrapper, torch::Tensor polyline_headers, torch::Tensor polygon_headers, torch::Tensor cubes, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold);
torch::Tensor ludus_render_fwd_cuda_ts(LudusCudaStateWrapper& stateWrapper, torch::Tensor polyline_headers, torch::Tensor polygon_headers, torch::Tensor cubes, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold, std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int, int>> cube_pool_list, std::vector<std::tuple<torch::Tensor, float, int64_t>> dot_list);
torch::Tensor ludus_render_fwd_cuda_timestamped(LudusCudaStateWrapper& stateWrapper, torch::Tensor timestamps, torch::Tensor int32_data, torch::Tensor vertices, torch::Tensor triangles, torch::Tensor float_data, torch::Tensor polyline_pools, torch::Tensor polygon_pools, torch::Tensor cube_pools, int64_t query_timestamp_us, int max_extrapolation_us, int max_varrays_per_ts_polyline, int max_varrays_per_ts_polygon, int camera_type_id, torch::Tensor camera_intrinsics, torch::Tensor camera_poses, std::tuple<int, int> resolution, float tessellation_threshold);

//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Ludus CUDA Context
    pybind11::class_<LudusCudaStateWrapper>(m, "LudusCudaStateWrapper").def(pybind11::init<int>())
        .def("set_line_widths",          &LudusCudaStateWrapper::setLineWidths)
        .def("set_resolution_scale",     &LudusCudaStateWrapper::setResolutionScale)
        .def("set_depth_scaling",        &LudusCudaStateWrapper::setDepthScaling)
        .def("set_cull_radius",          &LudusCudaStateWrapper::setCullRadius)
        .def("set_max_tessellation_levels", &LudusCudaStateWrapper::setMaxTessellationLevels)
        .def("upload_color_palette",     &LudusCudaStateWrapper::uploadColorPalette)
        .def("set_msaa_samples",         &LudusCudaStateWrapper::setMsaaSamples);

    // CudaRaster low-level API test wrapper
    pybind11::class_<CudaRasterTestWrapper>(m, "CudaRasterTestWrapper").def(pybind11::init<int>())
        .def("set_buffer_size",             &CudaRasterTestWrapper::setBufferSize)
        .def("set_viewport",                &CudaRasterTestWrapper::setViewport)
        .def("set_render_mode_flags",       &CudaRasterTestWrapper::setRenderModeFlags)
        .def("deferred_clear",              &CudaRasterTestWrapper::deferredClear)
        .def("set_vertex_buffer",           &CudaRasterTestWrapper::setVertexBuffer)
        .def("set_index_buffer",            &CudaRasterTestWrapper::setIndexBuffer)
        .def("set_tiebreaker_color_buffer", &CudaRasterTestWrapper::setTiebreakerColorBuffer)
        .def("set_deterministic_tiebreaker",&CudaRasterTestWrapper::setDeterministicTiebreaker)
        .def("draw_triangles",              &CudaRasterTestWrapper::drawTriangles)
        .def("swap_depth_and_peel",         &CudaRasterTestWrapper::swapDepthAndPeel)
        .def("get_color_buffer",            &CudaRasterTestWrapper::getColorBuffer)
        .def("get_depth_buffer",            &CudaRasterTestWrapper::getDepthBuffer)
        .def("get_buffer_width",            &CudaRasterTestWrapper::getBufferWidth)
        .def("get_buffer_height",           &CudaRasterTestWrapper::getBufferHeight)
        .def("get_num_images",              &CudaRasterTestWrapper::getNumImages);

    m.attr("CR_RENDER_MODE_ENABLE_BACKFACE_CULLING") =
        pybind11::int_((int)CR::CudaRaster::RenderModeFlag_EnableBackfaceCulling);
    m.attr("CR_RENDER_MODE_ENABLE_DEPTH_PEELING") =
        pybind11::int_((int)CR::CudaRaster::RenderModeFlag_EnableDepthPeeling);

    // CUDA rendering ops
    m.def("ludus_render_fwd_cuda", &ludus_render_fwd_cuda, "ludus f-theta CUDA rendering");
    m.def("ludus_render_fwd_cuda_ts", &ludus_render_fwd_cuda_ts, "ludus f-theta CUDA rendering with timestamped cube pools");
    m.def("ludus_render_fwd_cuda_timestamped", &ludus_render_fwd_cuda_timestamped, "ludus f-theta CUDA timestamped rendering from flat buffers");
}

//------------------------------------------------------------------------
