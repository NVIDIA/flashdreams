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

// Vulkan-backend Python bindings. Compiled into its own torch extension
// (ludus_renderer_vk_plugin) so the CUDA-only extension stays buildable on
// systems without Vulkan headers.

#include "torch_common.inl"
#include "../common/common.h"
#include "../render/ludus_vk.h"
#include <tuple>

//------------------------------------------------------------------------
// FLU (front-left-up) to RDF (right-down-front) basis conversion.
// The shaders consume RDF column-major matrices; PyTorch poses are FLU.
//------------------------------------------------------------------------

static torch::Tensor flu_to_rdf(const torch::Tensor& camera_poses)
{
    static const float kFluToRdf[16] = {
        0, -1,  0, 0,
        0,  0, -1, 0,
        1,  0,  0, 0,
        0,  0,  0, 1,
    };
    auto conv = torch::from_blob((void*)kFluToRdf, {4, 4}, torch::kFloat32)
                    .to(camera_poses.device());
    return torch::matmul(conv, camera_poses);
}

//------------------------------------------------------------------------
// Python-facing wrapper for the Vulkan timestamped renderer.
//------------------------------------------------------------------------

class LudusTimestampedVkStateWrapper
{
public:
    LudusTimestampedVkState*    pState;
    int                         cudaDeviceIdx;

    LudusTimestampedVkStateWrapper(int cudaDeviceIdx_)
    {
        pState = new LudusTimestampedVkState();
        cudaDeviceIdx = cudaDeviceIdx_;
        memset(pState, 0, sizeof(LudusTimestampedVkState));
        pState->cullRadiusScale = 1.5f;
        ludusTimestampedInitVk(NVDR_CTX_PARAMS, *pState, cudaDeviceIdx_);
    }

    ~LudusTimestampedVkStateWrapper()
    {
        ludusTimestampedReleaseVk(NVDR_CTX_PARAMS, *pState);
        delete pState;
    }

    int uploadScene(
        torch::Tensor scene_desc, torch::Tensor polyline_pools,
        torch::Tensor polygon_pools, torch::Tensor obstacle_pools,
        int max_obstacles_in_pool,
        int max_varrays_per_ts_polyline, int max_varrays_per_ts_polygon,
        torch::Tensor timestamps,
        torch::Tensor int32_data, torch::Tensor vertices,
        torch::Tensor triangles, torch::Tensor poses,
        torch::Tensor float_data)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(scene_desc));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        return ludusUploadSceneVk(
            NVDR_CTX_PARAMS, *pState, stream,
            reinterpret_cast<const TimestampedScene*>(scene_desc.data_ptr<uint8_t>()),
            reinterpret_cast<const TimestampedPolylinePool*>(polyline_pools.data_ptr<uint8_t>()),
            polyline_pools.numel() > 0 ? (int)polyline_pools.size(0) : 0,
            reinterpret_cast<const TimestampedPolygonPool*>(polygon_pools.data_ptr<uint8_t>()),
            polygon_pools.numel() > 0 ? (int)polygon_pools.size(0) : 0,
            reinterpret_cast<const ObstaclePool*>(obstacle_pools.data_ptr<uint8_t>()),
            obstacle_pools.numel() > 0 ? (int)obstacle_pools.size(0) : 0,
            max_obstacles_in_pool,
            max_varrays_per_ts_polyline, max_varrays_per_ts_polygon,
            timestamps.data_ptr<int64_t>(), (int)timestamps.numel(),
            int32_data.data_ptr<int32_t>(), (int)int32_data.numel(),
            reinterpret_cast<const Vertex*>(vertices.data_ptr<float>()), (int)vertices.size(0),
            reinterpret_cast<const Triangle*>(triangles.data_ptr<int32_t>()), (int)triangles.size(0),
            reinterpret_cast<const CameraPose*>(poses.data_ptr<float>()), (int)poses.size(0),
            float_data.data_ptr<float>(), (int)float_data.numel()
        );
    }

    void uploadCameras(torch::Tensor intrinsics)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(intrinsics));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ludusUploadCamerasVk(NVDR_CTX_PARAMS, *pState, stream,
            reinterpret_cast<const FThetaCamera*>(intrinsics.data_ptr<float>()),
            (int)intrinsics.size(0));
    }

    void uploadColorPalette(torch::Tensor colors)
    {
        // Accept either packed int32 RGBA8 (CUDA backend convention) or
        // float[N,4] RGBA in [0,1]. The Vulkan shaders sample float; we
        // convert int32 packed -> float here so callers can use either.
        const at::cuda::OptionalCUDAGuard device_guard(device_of(colors));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        torch::Tensor f;
        int numColors;
        if (colors.dtype() == torch::kInt32) {
            numColors = (int)colors.numel();
            auto packed = colors.to(torch::kInt64).clone();
            auto r = (packed.bitwise_and(0xFFLL)).to(torch::kFloat32).div(255.0f);
            auto g = (packed.bitwise_right_shift(8).bitwise_and(0xFFLL)).to(torch::kFloat32).div(255.0f);
            auto b = (packed.bitwise_right_shift(16).bitwise_and(0xFFLL)).to(torch::kFloat32).div(255.0f);
            auto a = (packed.bitwise_right_shift(24).bitwise_and(0xFFLL)).to(torch::kFloat32).div(255.0f);
            f = torch::stack({r, g, b, a}, /*dim=*/-1).contiguous();
        } else {
            TORCH_CHECK(colors.dtype() == torch::kFloat32,
                "upload_color_palette: expected int32 packed RGBA8 or float32 [N,4]");
            TORCH_CHECK(colors.dim() == 2 && colors.size(1) == 4,
                "upload_color_palette: float tensor must be [N,4]");
            f = colors.contiguous();
            numColors = (int)f.size(0);
        }
        // The tensor came in on CPU/CUDA; the VK uploader needs CUDA memory.
        if (!f.is_cuda())
            f = f.to(torch::kCUDA);

        ludusUploadColorPaletteVk(NVDR_CTX_PARAMS, *pState, stream,
            f.data_ptr<float>(), numColors);
    }

    void removeScene(int sceneId)
    {
        const at::cuda::OptionalCUDAGuard device_guard(c10::Device(c10::kCUDA, cudaDeviceIdx));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ludusRemoveSceneVk(NVDR_CTX_PARAMS, *pState, sceneId, stream);
    }

    void clearScenes() { ludusClearScenesVk(NVDR_CTX_PARAMS, *pState); }

    void setTessellationThreshold(float t) { pState->tessellationThreshold = t; }

    void setMaxTessellationLevels(int pl, int pg, int c) {
        pState->maxTessellationLevelPolyline = pl;
        pState->maxTessellationLevelPolygon = pg;
        pState->maxTessellationLevelCube = c;
    }

    void setLineWidths(float pr, float pb, float er, float eb, float w) {
        pState->widthPolylineRegular = pr;
        pState->widthPolylineBev = pb;
        pState->widthEgoTrajRegular = er;
        pState->widthEgoTrajBev = eb;
        pState->widthWireframe = w;
    }

    void setResolutionScale(float s) { pState->resolutionScale = s; }
    void setDepthScaling(float e) { pState->depthScaling = e; }
    void setCullRadius(float r) { pState->cullRadiusScale = r; }

    void setMsaaSamples(int s) {
        pState->msaaSamples = s;
        // Zero the cached dimensions so the next render re-enters
        // ensureFramebuffer, which rebuilds the framebuffer and (when the
        // sample count changed) the render pass and pipelines to match.
        pState->width = 0;
        pState->height = 0;
    }

    int getMaxBatchSize() {
        // Vulkan multi-view has driver-dependent layer limits; 2048 is a safe
        // working upper bound on contemporary NVIDIA drivers.
        return 2048;
    }
};

//------------------------------------------------------------------------
// Render batch
//------------------------------------------------------------------------

torch::Tensor ludus_timestamped_render_batch_vk(
    LudusTimestampedVkStateWrapper& stateWrapper,
    torch::Tensor queries, torch::Tensor camera_poses,
    std::tuple<int, int> resolution)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusTimestampedVkState& s = *stateWrapper.pState;

    NVDR_CHECK_DEVICE(queries, camera_poses);
    NVDR_CHECK_CONTIGUOUS(queries, camera_poses);

    int numQueries = queries.size(0);
    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);

    // FLU -> RDF then transpose (column-major mat4 for GLSL).
    torch::Tensor poses_rdf = flu_to_rdf(camera_poses).transpose(-2, -1).contiguous();

    ludusRenderBatchVk(NVDR_CTX_PARAMS, s, stream,
        reinterpret_cast<const RenderQuery*>(queries.data_ptr<uint8_t>()),
        reinterpret_cast<const CameraPose*>(poses_rdf.data_ptr<float>()),
        numQueries, width, height);

    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(queries.device());
    torch::Tensor out = torch::empty({numQueries, height, width, 4}, opts);

    ludusCopyBatchResultsVk(NVDR_CTX_PARAMS, s, stream,
        out.data_ptr<uint8_t>(), width, height, numQueries);

    return out;
}

//------------------------------------------------------------------------
// Render to staging (for double-buffered async output)
//------------------------------------------------------------------------

std::tuple<int, bool> ludus_timestamped_render_to_staging_vk(
    LudusTimestampedVkStateWrapper& stateWrapper,
    torch::Tensor queries, torch::Tensor camera_poses,
    std::tuple<int, int> resolution)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    LudusTimestampedVkState& s = *stateWrapper.pState;

    NVDR_CHECK_DEVICE(queries, camera_poses);
    NVDR_CHECK_CONTIGUOUS(queries, camera_poses);

    int numQueries = queries.size(0);
    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);

    torch::Tensor poses_rdf = flu_to_rdf(camera_poses).transpose(-2, -1).contiguous();

    ludusRenderBatchVk(NVDR_CTX_PARAMS, s, stream,
        reinterpret_cast<const RenderQuery*>(queries.data_ptr<uint8_t>()),
        reinterpret_cast<const CameraPose*>(poses_rdf.data_ptr<float>()),
        numQueries, width, height);

    int stagingIdx = ludusCopyBatchResultsToStagingVk(NVDR_CTX_PARAMS, s, stream,
        width, height, numQueries);

    return std::make_tuple(stagingIdx, true);
}

torch::Tensor ludus_timestamped_get_staging_data_vk(
    LudusTimestampedVkStateWrapper& stateWrapper, int stagingIdx)
{
    LudusTimestampedVkState& s = *stateWrapper.pState;
    int w = s.stagingWidth, h = s.stagingHeight, n = s.stagingNumQueries;
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, stateWrapper.cudaDeviceIdx);
    torch::Tensor out = torch::empty({n, h, w, 4}, opts);

    ludusCopyStagingToOutputVk(NVDR_CTX_PARAMS, s, stagingIdx,
        out.data_ptr<uint8_t>(), w, h, n);
    return out;
}

py::list ludus_timestamped_encode_jpeg_batch_staging_vk(
    LudusTimestampedVkStateWrapper& stateWrapper, int stagingIdx, int quality)
{
    LudusTimestampedVkState& s = *stateWrapper.pState;
    std::vector<std::pair<uint8_t*, size_t>> jpegs;
    ludusEncodeJpegBatchStagingVk(NVDR_CTX_PARAMS, s, stagingIdx, quality, jpegs);

    py::list result;
    for (auto& [data, size] : jpegs) {
        result.append(py::bytes(reinterpret_cast<char*>(data), size));
        free(data);
    }
    return result;
}

bool ludus_timestamped_is_nvjpeg_available_vk(LudusTimestampedVkStateWrapper& /*stateWrapper*/)
{
    return true;
}

//------------------------------------------------------------------------
// pybind11 module
//------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<LudusTimestampedVkStateWrapper>(m, "LudusTimestampedVkStateWrapper")
        .def(pybind11::init<int>())
        .def("upload_scene",                  &LudusTimestampedVkStateWrapper::uploadScene)
        .def("upload_cameras",                &LudusTimestampedVkStateWrapper::uploadCameras)
        .def("upload_color_palette",          &LudusTimestampedVkStateWrapper::uploadColorPalette)
        .def("remove_scene",                  &LudusTimestampedVkStateWrapper::removeScene)
        .def("clear_scenes",                  &LudusTimestampedVkStateWrapper::clearScenes)
        .def("set_tessellation_threshold",    &LudusTimestampedVkStateWrapper::setTessellationThreshold)
        .def("set_max_tessellation_levels",   &LudusTimestampedVkStateWrapper::setMaxTessellationLevels)
        .def("set_line_widths",               &LudusTimestampedVkStateWrapper::setLineWidths)
        .def("set_resolution_scale",          &LudusTimestampedVkStateWrapper::setResolutionScale)
        .def("set_depth_scaling",             &LudusTimestampedVkStateWrapper::setDepthScaling)
        .def("set_cull_radius",               &LudusTimestampedVkStateWrapper::setCullRadius)
        .def("set_msaa_samples",              &LudusTimestampedVkStateWrapper::setMsaaSamples)
        .def("get_max_batch_size",            &LudusTimestampedVkStateWrapper::getMaxBatchSize);

    m.def("ludus_timestamped_render_batch",            &ludus_timestamped_render_batch_vk);
    m.def("ludus_timestamped_render_to_staging",       &ludus_timestamped_render_to_staging_vk);
    m.def("ludus_timestamped_get_staging_data",        &ludus_timestamped_get_staging_data_vk);
    m.def("ludus_timestamped_is_nvjpeg_available",     &ludus_timestamped_is_nvjpeg_available_vk);
    m.def("ludus_timestamped_encode_jpeg_batch_staging", &ludus_timestamped_encode_jpeg_batch_staging_vk);
}
