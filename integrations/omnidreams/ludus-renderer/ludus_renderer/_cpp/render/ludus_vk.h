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

#if !(defined(NVDR_TORCH) && defined(__CUDACC__))
#include "../common/framework.h"
#include "../common/vkutil.h"
#include "ludus_types.h"
#include <nvjpeg.h>
#include <vector>

// Forward-declared helper kernel (defined in render/ludus_jpeg.cu).
extern "C" void launchRgbaToRgbFlip(
    const uint8_t* srcRgba, uint8_t* dstRgb,
    int width, int height, cudaStream_t stream
);

// ========================================================================
// Vulkan Timestamped Rendering State
//
// Mirrors the GL-based timestamped renderer (now removed from this project)
// but uses Vulkan task/mesh/fragment pipelines via VK_EXT_mesh_shader.
//
// Same public API signatures as the CUDA backend in ludus_cuda.h so the
// Python bindings can dispatch to either with minimal differences.
// ========================================================================

struct LudusTimestampedVkState
{
    // ---------- Framebuffer ----------
    int                     width;
    int                     height;
    int                     maxLayers;

    // ---------- Scene storage bookkeeping ----------
    int                     maxScenes;
    int                     numScenes;

    int                     timestampsCapacity, timestampsUsed;
    int                     int32Capacity, int32Used;
    int                     vertexCapacity, vertexUsed;
    int                     triangleCapacity, triangleUsed;
    int                     poseCapacity, poseUsed;
    int                     floatCapacity, floatUsed;
    int                     polylinePoolCapacity, polylinePoolUsed;
    int                     polygonPoolCapacity, polygonPoolUsed;
    int                     obstaclePoolCapacity, obstaclePoolUsed;
    int                     maxObstaclesPerPool;
    int                     maxCubePoolsPerScene;
    int                     maxPolylinePoolsPerScene;
    int                     maxPolygonPoolsPerScene;
    // Max varrays in any pool at a single timestamp, per family. Drives the
    // per-pool mesh-task dispatch stride so large pools aren't silently capped.
    int                     maxVarraysPerTsPolyline;
    int                     maxVarraysPerTsPolygon;

    // ---------- Cameras ----------
    int                     cameraCapacity;
    int                     numCameras;

    // ---------- Query batch ----------
    int                     queryCapacity;

    // ---------- Vulkan context ----------
    VkContext               vkctx;

    // ---------- Framebuffer images ----------
    VkExternalImage         colorImage;
    VkExternalImage         depthStencilImage;

    // ---------- MSAA ----------
    int                     msaaSamples;
    VkExternalImage         colorImageMSAA;
    VkExternalImage         depthStencilImageMSAA;

    // ---------- Render pass and framebuffer ----------
    VkRenderPass            renderPass;
    VkFramebuffer           framebuffer;
    // Sample count the renderPass + pipelines were last built with (1 = no
    // MSAA). The render pass attachment layout and the pipelines bake in the
    // sample count, so both must be rebuilt when msaaSamples changes.
    int                     renderPassSamples;

    // ---------- Data buffers (SSBOs, CUDA-importable) ----------
    VkExternalBuffer        timestampsBuffer;       // binding 0:  int64[]
    VkExternalBuffer        int32Buffer;            // binding 1:  int32[]
    VkExternalBuffer        vertexBuffer;           // binding 2:  Vertex[]
    VkExternalBuffer        triangleBuffer;         // binding 3:  Triangle[]
    VkExternalBuffer        poseBuffer;             // binding 4:  CameraPose[]
    VkExternalBuffer        floatBuffer;            // binding 5:  float[]
    VkExternalBuffer        sceneBuffer;            // binding 6:  TimestampedScene[]
    VkExternalBuffer        polylinePoolBuffer;     // binding 7:  TimestampedPolylinePool[]
    VkExternalBuffer        polygonPoolBuffer;      // binding 8:  TimestampedPolygonPool[]
    VkExternalBuffer        obstaclePoolBuffer;     // binding 9:  ObstaclePool[]
    VkExternalBuffer        colorPaletteBuffer;     // binding 10: vec4[]
    VkExternalBuffer        cameraIntrinsicsBuffer; // binding 11: FThetaCamera[]
    VkExternalBuffer        cameraPoseBuffer;       // binding 12: CameraPose[] per-query
    VkExternalBuffer        queryBuffer;            // binding 13: RenderQuery[]

    int                     colorPaletteSize;

    // Set by control-plane ops (upload/remove/clear); the per-frame host
    // roundtrip re-pushes the scene SSBOs only when set.
    int                     sceneBuffersDirty;

    // ---------- Descriptor set ----------
    VkDescriptorSetLayout   descriptorSetLayout;
    VkDescriptorPool        descriptorPool;
    VkDescriptorSet         descriptorSet;

    // ---------- Pipeline layout (shared across all 3 pipelines) ----------
    VkPipelineLayout        pipelineLayout;

    // ---------- Pipelines (one per draw type) ----------
    VkPipeline              pipelinePolyline;
    VkPipeline              pipelinePolygon;
    VkPipeline              pipelineObstacle;

    // ---------- Shader modules: 3 pipelines x 3 stages (task, mesh, frag) ----------
    VkShaderModule          shaderModules[9];

    // ---------- Capability flags ----------
    int                     hasMeshShader;
    float                   tessellationThreshold;

    // ---------- Configurable parameters ----------
    int                     maxTessellationLevelPolyline;
    int                     maxTessellationLevelPolygon;
    int                     maxTessellationLevelCube;
    float                   widthPolylineRegular;
    float                   widthPolylineBev;
    float                   widthEgoTrajRegular;
    float                   widthEgoTrajBev;
    float                   widthWireframe;
    float                   resolutionScale;
    float                   depthScaling;
    int                     maxExtrapolationUs;
    float                   cullRadiusScale;

    // ---------- Double-buffered staging for async transfer ----------
    uint8_t*                stagingBuffer[2];
    size_t                  stagingBufferSize;
    int                     stagingWidth, stagingHeight, stagingNumQueries;
    int                     currentStagingIdx;
    int                     stagingValid[2];
    cudaStream_t            copyStream;
    cudaEvent_t             stagingReadyEvent[2];

    uint8_t*                pinnedHostBuffer[2];
    size_t                  pinnedHostBufferSize;
    int                     currentPinnedIdx;
    int                     pinnedValid[2];
    int                     pinnedWidth[2], pinnedHeight[2], pinnedNumQueries[2];
    cudaEvent_t             pinnedReadyEvent[2];

    // ---------- NVJPEG (GPU JPEG encoding) ----------
    nvjpegHandle_t          nvjpegHandle;
    nvjpegEncoderState_t    nvjpegEncoderState;
    nvjpegEncoderParams_t   nvjpegEncoderParams;
    int                     nvjpegInitialized;
    uint8_t*                jpegOutputBuffer;
    size_t                  jpegOutputBufferSize;
    uint8_t*                jpegFlipBuffer;
    size_t                  jpegFlipBufferSize;

    // ---------- Per-state readback buffer (image -> CUDA tensor) ----------
    // Must be a per-state member (not a process-wide static) so the
    // Vulkan handles stay tied to a single VkDevice's lifetime, which
    // matters when multiple LudusTimestampedContext objects are created
    // and destroyed in the same process.
    VkExternalBuffer        readbackBuffer;
};

// Unified push constants block, shared across all 3 pipelines and stages.
// 21 fields, 84 bytes -- must match the layout(push_constant) block in
// every Vulkan GLSL shader. Per-cube wireframe rendering is driven by the
// ObstaclePool.render_flags SSBO field (not a push constant) so that
// different cube pools in the same draw can mix solid and wireframe modes.
struct LudusPushConstants
{
    float    u_width_polyline_regular;     // 0
    float    u_width_polyline_bev;         // 4
    float    u_width_ego_traj_regular;     // 8
    float    u_width_ego_traj_bev;         // 12
    float    u_width_wireframe;            // 16
    float    u_resolution_scale;           // 20
    float    u_depth_scaling;              // 24
    int32_t  u_max_extrapolation_us;       // 28
    int32_t  u_color_palette_size;         // 32
    uint32_t u_num_queries;                // 36
    float    u_tessellation_threshold;     // 40
    uint32_t u_max_tessellation_polyline;  // 44
    uint32_t u_max_tessellation_polygon;   // 48
    uint32_t u_max_tessellation_cube;      // 52
    float    u_cull_radius_scale;          // 56
    float    u_fog_enabled;                // 60
    uint32_t u_max_obstacles;              // 64
    uint32_t u_cube_pool_index;            // 68
    uint32_t u_num_polygon_pools;          // 72
    uint32_t u_max_varrays_per_pool;       // 76
    uint32_t u_num_polyline_pools;         // 80
};
static_assert(sizeof(LudusPushConstants) == 84, "LudusPushConstants must be 84 bytes");

// ========================================================================
// Public API -- same shape as the GL/CUDA backends so bindings can dispatch.
// ========================================================================

void ludusTimestampedInitVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s, int cudaDeviceIdx);

void ludusUploadCamerasVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const FThetaCamera* intrinsics, int numCameras
);

void ludusUploadColorPaletteVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const float* colors, int numColors
);

int ludusUploadSceneVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const TimestampedScene* sceneDesc,
    const TimestampedPolylinePool* polylinePools, int numPolylinePools,
    const TimestampedPolygonPool* polygonPools, int numPolygonPools,
    const ObstaclePool* obstaclePools, int numObstaclePools,
    int maxObstaclesInPool,
    int maxVarraysPerTsPolyline, int maxVarraysPerTsPolygon,
    const int64_t* timestamps, int numTimestamps,
    const int32_t* int32Data, int numInt32,
    const Vertex* vertices, int numVertices,
    const Triangle* triangles, int numTriangles,
    const CameraPose* poses, int numPoses,
    const float* floatData, int numFloats
);

void ludusRemoveSceneVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s, int sceneId, cudaStream_t stream);

void ludusRenderBatchVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const RenderQuery* queries, const CameraPose* cameraPoses,
    int numQueries, int width, int height
);

void ludusCopyBatchResultsVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    uint8_t* outputPtr, int width, int height, int numQueries
);

int ludusCopyBatchResultsToStagingVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    int width, int height, int numQueries
);

void ludusCopyStagingToOutputVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx,
    uint8_t* outputPtr, int width, int height, int numQueries
);

int ludusStartAsyncHostTransferVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx
);

int ludusIsPinnedBufferReadyVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int pinnedIdx
);

int ludusIsHostTransferCompleteVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s
);

int ludusEncodeJpegBatchStagingVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx, int quality,
    std::vector<std::pair<uint8_t*, size_t>>& outJpegs
);

void ludusClearScenesVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s);
void ludusTimestampedReleaseVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s);

#endif // !(defined(NVDR_TORCH) && defined(__CUDACC__))
