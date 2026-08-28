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
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
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

struct VulkanSharedState
{
    enum OutputSlotState : int
    {
        OutputSlotFree = 0,
        OutputSlotLeased,
        OutputSlotPendingRelease,
        OutputSlotReclaiming,
    };

    LudusTimestampedVkState* pState;
    std::array<std::atomic<int>, LUDUS_VK_OUTPUT_SLOTS> outputSlotStates;
    int cudaDeviceIdx;

    explicit VulkanSharedState(int deviceIdx)
        : pState(new LudusTimestampedVkState())
        , cudaDeviceIdx(deviceIdx)
    {
        const at::cuda::OptionalCUDAGuard device_guard(
            c10::Device(c10::kCUDA, cudaDeviceIdx));
        memset(pState, 0, sizeof(LudusTimestampedVkState));
        for (int i = 0; i < LUDUS_VK_OUTPUT_SLOTS; ++i) {
            outputSlotStates[i].store(
                OutputSlotFree, std::memory_order_relaxed);
        }
        ludusTimestampedInitVk(NVDR_CTX_PARAMS, *pState, cudaDeviceIdx);
    }

    ~VulkanSharedState()
    {
        cudaSetDevice(cudaDeviceIdx);
        // Output tensors keep this owner alive. Their final references can be
        // dropped before arbitrary CUDA consumer streams finish, so teardown
        // must drain the device before destroying the imported allocations.
        cudaDeviceSynchronize();
        ludusTimestampedReleaseVk(NVDR_CTX_PARAMS, *pState);
        delete pState;
    }
};

class LudusTimestampedVkStateWrapper
{
public:
    std::shared_ptr<VulkanSharedState> sharedState;
    LudusTimestampedVkState*    pState;
    int                         cudaDeviceIdx;
    cudaEvent_t                 lastUseEvent;
    bool                        hasLastUseEvent;
    std::mutex                  stateMutex;
    int                         outputCursor;

    LudusTimestampedVkStateWrapper(int cudaDeviceIdx_)
        : sharedState(std::make_shared<VulkanSharedState>(cudaDeviceIdx_))
        , pState(sharedState->pState)
        , cudaDeviceIdx(cudaDeviceIdx_)
        , lastUseEvent(nullptr)
        , hasLastUseEvent(false)
        , outputCursor(0)
    {
        const at::cuda::OptionalCUDAGuard device_guard(
            c10::Device(c10::kCUDA, cudaDeviceIdx));
        AT_CUDA_CHECK(cudaEventCreateWithFlags(&lastUseEvent, cudaEventDisableTiming));
    }

    ~LudusTimestampedVkStateWrapper()
    {
        const at::cuda::OptionalCUDAGuard device_guard(
            c10::Device(c10::kCUDA, cudaDeviceIdx));
        std::lock_guard<std::mutex> lock(stateMutex);
        if (hasLastUseEvent)
            AT_CUDA_CHECK(cudaEventSynchronize(lastUseEvent));
        AT_CUDA_CHECK(cudaEventDestroy(lastUseEvent));
    }

    void waitForLastUse(cudaStream_t stream)
    {
        if (hasLastUseEvent)
            AT_CUDA_CHECK(cudaStreamWaitEvent(stream, lastUseEvent, 0));
    }

    void recordLastUse(cudaStream_t stream)
    {
        AT_CUDA_CHECK(cudaEventRecord(lastUseEvent, stream));
        hasLastUseEvent = true;
    }

    int acquireStagingSlot()
    {
        for (int attempt = 0; attempt < 2; ++attempt) {
            const int slot = (pState->currentStagingIdx + attempt) % 2;
            if (!pState->stagingValid[slot]) {
                // Reserve before submitting any work. stateMutex serializes
                // acquisition and consumption, so no atomic is needed here.
                pState->stagingValid[slot] = 1;
                pState->currentStagingIdx = (slot + 1) % 2;
                return slot;
            }
        }
        return -1;
    }

    void releaseStagingSlot(int slot)
    {
        pState->stagingValid[slot] = 0;
    }

    int acquireOutputSlot()
    {
        auto tryAcquireFreeSlot = [&]() {
            for (int attempt = 0; attempt < LUDUS_VK_OUTPUT_SLOTS; ++attempt) {
                int slot = (outputCursor + attempt) % LUDUS_VK_OUTPUT_SLOTS;
                int expected = VulkanSharedState::OutputSlotFree;
                if (sharedState->outputSlotStates[slot].compare_exchange_strong(
                        expected, VulkanSharedState::OutputSlotLeased,
                        std::memory_order_acq_rel)) {
                    outputCursor = (slot + 1) % LUDUS_VK_OUTPUT_SLOTS;
                    return slot;
                }
            }
            return -1;
        };

        int slot = tryAcquireFreeSlot();
        if (slot >= 0)
            return slot;

        // A from_blob allocation is outside PyTorch's CUDA caching allocator,
        // so record_stream cannot report every stream that consumed it. Move a
        // stable snapshot of dropped tensors into a reclaiming state, then one
        // batched device drain makes all of those slots safe to reuse. This is
        // paid only when the pool is exhausted, never once per returned frame.
        std::array<int, LUDUS_VK_OUTPUT_SLOTS> reclaimSlots;
        int reclaimCount = 0;
        for (int i = 0; i < LUDUS_VK_OUTPUT_SLOTS; ++i) {
            int expected = VulkanSharedState::OutputSlotPendingRelease;
            if (sharedState->outputSlotStates[i].compare_exchange_strong(
                    expected, VulkanSharedState::OutputSlotReclaiming,
                    std::memory_order_acq_rel)) {
                reclaimSlots[reclaimCount++] = i;
            }
        }
        if (reclaimCount > 0) {
            cudaError_t syncError = cudaDeviceSynchronize();
            if (syncError != cudaSuccess) {
                for (int i = 0; i < reclaimCount; ++i) {
                    sharedState->outputSlotStates[reclaimSlots[i]].store(
                        VulkanSharedState::OutputSlotPendingRelease,
                        std::memory_order_release);
                }
                TORCH_CHECK(false,
                    "failed to reclaim Vulkan interop export slots: ",
                    cudaGetErrorString(syncError));
            }
            for (int i = 0; i < reclaimCount; ++i) {
                sharedState->outputSlotStates[reclaimSlots[i]].store(
                    VulkanSharedState::OutputSlotFree,
                    std::memory_order_release);
            }
            slot = tryAcquireFreeSlot();
            TORCH_INTERNAL_ASSERT(slot >= 0);
            return slot;
        }

        TORCH_CHECK(false,
            "all Vulkan interop export slots are still referenced; "
            "release an older render tensor before submitting another batch");
        return -1;
    }

    void releaseOutputSlot(int slot)
    {
        sharedState->outputSlotStates[slot].store(
            VulkanSharedState::OutputSlotFree, std::memory_order_release);
    }

    void quarantineOutputSlot(int slot)
    {
        sharedState->outputSlotStates[slot].store(
            VulkanSharedState::OutputSlotPendingRelease,
            std::memory_order_release);
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
        std::lock_guard<std::mutex> lock(stateMutex);
        waitForLastUse(stream);
        int sceneId = ludusUploadSceneVk(
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
        recordLastUse(stream);
        return sceneId;
    }

    void uploadCameras(torch::Tensor intrinsics)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(intrinsics));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        std::lock_guard<std::mutex> lock(stateMutex);
        waitForLastUse(stream);
        ludusUploadCamerasVk(NVDR_CTX_PARAMS, *pState, stream,
            reinterpret_cast<const FThetaCamera*>(intrinsics.data_ptr<float>()),
            (int)intrinsics.size(0));
        recordLastUse(stream);
    }

    void uploadColorPalette(torch::Tensor colors)
    {
        // Accept either packed int32 RGBA8 (CUDA backend convention) or
        // float[N,4] RGBA in [0,1]. The Vulkan shaders sample float; we
        // convert int32 packed -> float here so callers can use either.
        const at::cuda::OptionalCUDAGuard device_guard(device_of(colors));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        std::lock_guard<std::mutex> lock(stateMutex);
        waitForLastUse(stream);

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
        recordLastUse(stream);
    }

    void removeScene(int sceneId)
    {
        const at::cuda::OptionalCUDAGuard device_guard(c10::Device(c10::kCUDA, cudaDeviceIdx));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        std::lock_guard<std::mutex> lock(stateMutex);
        waitForLastUse(stream);
        ludusRemoveSceneVk(NVDR_CTX_PARAMS, *pState, sceneId, stream);
        recordLastUse(stream);
    }

    void updateCubePool(
        int absolutePoolIndex,
        int timestampOffset, int trackTimestampOffset, int int32Offset,
        int translationOffset, int quaternionOffset, int scaleOffset, int colorOffset,
        torch::Tensor timestamps, torch::Tensor trackTimestamps,
        torch::Tensor prefixSum, torch::Tensor translations,
        torch::Tensor quaternions, torch::Tensor scales,
        torch::Tensor colors, torch::Tensor poolHeader)
    {
        const at::cuda::OptionalCUDAGuard device_guard(device_of(timestamps));
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        std::lock_guard<std::mutex> lock(stateMutex);
        waitForLastUse(stream);

        NVDR_CHECK_DEVICE(timestamps, trackTimestamps, prefixSum, translations,
            quaternions, scales, colors, poolHeader);
        NVDR_CHECK_CONTIGUOUS(timestamps, trackTimestamps, prefixSum, translations,
            quaternions, scales, colors, poolHeader);
        TORCH_CHECK(timestamps.dtype() == torch::kInt64 &&
            trackTimestamps.dtype() == torch::kInt64,
            "update_cube_pool timestamps must be int64");
        TORCH_CHECK(prefixSum.dtype() == torch::kInt32,
            "update_cube_pool prefix sum must be int32");
        NVDR_CHECK_F32(translations, quaternions, scales, colors);
        TORCH_CHECK(poolHeader.dtype() == torch::kUInt8 && poolHeader.numel() == 64,
            "update_cube_pool header must contain exactly 64 uint8 values");

        ludusUpdateCubePoolVk(
            NVDR_CTX_PARAMS, *pState, stream, absolutePoolIndex,
            timestampOffset, trackTimestampOffset, int32Offset,
            translationOffset, quaternionOffset, scaleOffset, colorOffset,
            timestamps.data_ptr<int64_t>(), (int)timestamps.numel(),
            trackTimestamps.data_ptr<int64_t>(), (int)trackTimestamps.numel(),
            prefixSum.data_ptr<int32_t>(), (int)prefixSum.numel(),
            translations.data_ptr<float>(), (int)translations.numel(),
            quaternions.data_ptr<float>(), (int)quaternions.numel(),
            scales.data_ptr<float>(), (int)scales.numel(),
            colors.data_ptr<float>(), (int)colors.numel(),
            reinterpret_cast<const ObstaclePool*>(poolHeader.data_ptr<uint8_t>()));
        recordLastUse(stream);
    }

    void clearScenes() {
        std::lock_guard<std::mutex> lock(stateMutex);
        ludusClearScenesVk(NVDR_CTX_PARAMS, *pState);
    }

    void setTessellationThreshold(float t) {
        std::lock_guard<std::mutex> lock(stateMutex);
        pState->tessellationThreshold = t;
    }

    void setMaxTessellationLevels(int pl, int pg, int c) {
        std::lock_guard<std::mutex> lock(stateMutex);
        pState->maxTessellationLevelPolyline = pl;
        pState->maxTessellationLevelPolygon = pg;
        pState->maxTessellationLevelCube = c;
    }

    void setLineWidths(float pr, float pb, float er, float eb, float w) {
        std::lock_guard<std::mutex> lock(stateMutex);
        pState->widthPolylineRegular = pr;
        pState->widthPolylineBev = pb;
        pState->widthEgoTrajRegular = er;
        pState->widthEgoTrajBev = eb;
        pState->widthWireframe = w;
    }

    void setResolutionScale(float s) {
        std::lock_guard<std::mutex> lock(stateMutex);
        pState->resolutionScale = s;
    }
    void setDepthScaling(float e) {
        std::lock_guard<std::mutex> lock(stateMutex);
        pState->depthScaling = e;
    }
    void setCullRadius(float r) {
        std::lock_guard<std::mutex> lock(stateMutex);
        pState->cullRadiusScale = r;
    }

    void setMsaaSamples(int s) {
        std::lock_guard<std::mutex> lock(stateMutex);
        TORCH_CHECK(s <= 1,
            "Vulkan color-attachment compute export currently supports one "
            "sample; multisample resolve is not enabled");
        pState->msaaSamples = s;
        // ensureFramebuffer detects the sample mismatch and rebuilds the
        // depth-target cache and compatible pipelines on the next render.
    }

    int getMaxBatchSize() {
        std::lock_guard<std::mutex> lock(stateMutex);
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
    std::lock_guard<std::mutex> state_lock(stateWrapper.stateMutex);
    stateWrapper.waitForLastUse(stream);
    LudusTimestampedVkState& s = *stateWrapper.pState;

    NVDR_CHECK_DEVICE(queries, camera_poses);
    NVDR_CHECK_CONTIGUOUS(queries, camera_poses);

    int numQueries = queries.size(0);
    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(queries.device());
    if (numQueries == 0)
        return torch::empty({0, height, width, 4}, opts);

    // FLU -> RDF then transpose (column-major mat4 for GLSL).
    torch::Tensor poses_rdf = flu_to_rdf(camera_poses).transpose(-2, -1).contiguous();
    const int outputSlot = stateWrapper.acquireOutputSlot();
    try {
        ludusRenderBatchVk(NVDR_CTX_PARAMS, s, stream,
            reinterpret_cast<const RenderQuery*>(queries.data_ptr<uint8_t>()),
            reinterpret_cast<const CameraPose*>(poses_rdf.data_ptr<float>()),
            numQueries, width, height, outputSlot);

        uint8_t* sharedOutput = ludusMapBatchResultsVk(
            NVDR_CTX_PARAMS, s, stream, outputSlot,
            width, height, numQueries);
        auto sharedState = stateWrapper.sharedState;
        torch::Tensor out = torch::from_blob(
            sharedOutput,
            {numQueries, height, width, 4},
            [sharedState, outputSlot](void*) {
                // Final CPU ownership does not imply completion on every CUDA
                // stream that may have consumed this external allocation.
                // Quarantine it; acquireOutputSlot reclaims dropped slots only
                // after a batched device-wide completion check.
                sharedState->outputSlotStates[outputSlot].store(
                    VulkanSharedState::OutputSlotPendingRelease,
                    std::memory_order_release);
            },
            opts);
        stateWrapper.recordLastUse(stream);
        return out;
    } catch (...) {
        stateWrapper.quarantineOutputSlot(outputSlot);
        throw;
    }
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
    std::lock_guard<std::mutex> state_lock(stateWrapper.stateMutex);
    stateWrapper.waitForLastUse(stream);
    LudusTimestampedVkState& s = *stateWrapper.pState;

    NVDR_CHECK_DEVICE(queries, camera_poses);
    NVDR_CHECK_CONTIGUOUS(queries, camera_poses);

    int numQueries = queries.size(0);
    int height = std::get<0>(resolution);
    int width = std::get<1>(resolution);

    torch::Tensor poses_rdf = flu_to_rdf(camera_poses).transpose(-2, -1).contiguous();
    const int stagingIdx = stateWrapper.acquireStagingSlot();
    if (stagingIdx < 0)
        return std::make_tuple(-1, false);

    int outputSlot = -1;
    try {
        outputSlot = stateWrapper.acquireOutputSlot();
        ludusRenderBatchVk(NVDR_CTX_PARAMS, s, stream,
            reinterpret_cast<const RenderQuery*>(queries.data_ptr<uint8_t>()),
            reinterpret_cast<const CameraPose*>(poses_rdf.data_ptr<float>()),
            numQueries, width, height, outputSlot);

        ludusCopyBatchResultsToStagingVk(
            NVDR_CTX_PARAMS, s, stream, outputSlot, stagingIdx,
            width, height, numQueries);
        stateWrapper.recordLastUse(stream);
        // The staging copy is ordered before lastUseEvent, so the next wrapper
        // call cannot reuse this slot until that copy has completed.
        stateWrapper.releaseOutputSlot(outputSlot);
        return std::make_tuple(stagingIdx, true);
    } catch (...) {
        if (outputSlot >= 0)
            stateWrapper.quarantineOutputSlot(outputSlot);
        stateWrapper.releaseStagingSlot(stagingIdx);
        throw;
    }
}

torch::Tensor ludus_timestamped_get_staging_data_vk(
    LudusTimestampedVkStateWrapper& stateWrapper, int stagingIdx)
{
    std::lock_guard<std::mutex> state_lock(stateWrapper.stateMutex);
    LudusTimestampedVkState& s = *stateWrapper.pState;
    TORCH_CHECK(stagingIdx >= 0 && stagingIdx < 2,
        "invalid Vulkan staging slot ", stagingIdx);
    TORCH_CHECK(s.stagingValid[stagingIdx],
        "Vulkan staging slot ", stagingIdx, " has no pending frame");
    int w = s.stagingWidth[stagingIdx];
    int h = s.stagingHeight[stagingIdx];
    int n = s.stagingNumQueries[stagingIdx];
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, stateWrapper.cudaDeviceIdx);
    torch::Tensor out = torch::empty({n, h, w, 4}, opts);

    ludusCopyStagingToOutputVk(
        NVDR_CTX_PARAMS, s, stagingIdx, out.data_ptr<uint8_t>());
    stateWrapper.releaseStagingSlot(stagingIdx);
    return out;
}

py::list ludus_timestamped_encode_jpeg_batch_staging_vk(
    LudusTimestampedVkStateWrapper& stateWrapper, int stagingIdx, int quality)
{
    std::lock_guard<std::mutex> state_lock(stateWrapper.stateMutex);
    LudusTimestampedVkState& s = *stateWrapper.pState;
    std::vector<std::pair<uint8_t*, size_t>> jpegs;
    ludusEncodeJpegBatchStagingVk(NVDR_CTX_PARAMS, s, stagingIdx, quality, jpegs);

    py::list result;
    for (auto& [data, size] : jpegs) {
        result.append(py::bytes(reinterpret_cast<char*>(data), size));
        free(data);
    }
    stateWrapper.releaseStagingSlot(stagingIdx);
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
        .def("update_cube_pool",              &LudusTimestampedVkStateWrapper::updateCubePool)
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
