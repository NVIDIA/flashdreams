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

//=============================================================================
// Vulkan Timestamped Renderer (VK_EXT_mesh_shader path).
//
// Port from the GL-based timestamped renderer to native Vulkan. Geometry is
// generated procedurally in task/mesh shaders. CUDA tensors are bridged in
// through platform-native Vulkan external memory: every SSBO that needs to
// receive CUDA writes is imported into CUDA, allowing scene/query writes into
// Vulkan buffers. Fixed-function color/depth rendering resolves visibility,
// then compute writes the result into a CUDA-visible SSBO mapped directly by
// PyTorch without a host or CUDA result copy.
//=============================================================================

#include "ludus_vk.h"
#include "shaders_spv.h"   // generated header with embedded SPIR-V byte arrays
#include <cstring>
#include <algorithm>
#include <climits>
#include <fstream>
#include <vector>

// VK_CHECK / VK_DBG / ludus_vk_debug() are shared from vkutil.h (included via
// ludus_vk.h).

// Vulkan requires every vkCmdPushConstants update to include all stages from
// each overlapping range in the pipeline layout. Graphics and compute share
// this one struct/range, so use one mask for both declaration and updates.
static constexpr VkShaderStageFlags kLudusPushConstantStages =
    VK_SHADER_STAGE_TASK_BIT_EXT
    | VK_SHADER_STAGE_MESH_BIT_EXT
    | VK_SHADER_STAGE_FRAGMENT_BIT
    | VK_SHADER_STAGE_COMPUTE_BIT;

//=============================================================================
// SPIR-V Shader Module Creation
//=============================================================================

static VkShaderModule createShaderModule(VkDevice device, const uint32_t* code, size_t bytes)
{
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &module));
    return module;
}

//=============================================================================
// Render Pass
//=============================================================================

static VkRenderPass createTimestampedRenderPass(
    VkDevice device, VkFormat depthFormat, VkSampleCountFlagBits samples)
{
    VkAttachmentDescription color = {};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.samples = samples;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkAttachmentDescription depth = {};
    depth.format = depthFormat;
    depth.samples = samples;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = {
        0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef = {
        1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep = {};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                     | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                     | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                      | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = {color, depth};

    VkRenderPassCreateInfo rpCI = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpCI.attachmentCount = 2;
    rpCI.pAttachments = attachments;
    rpCI.subpassCount = 1;
    rpCI.pSubpasses = &subpass;
    rpCI.dependencyCount = 1;
    rpCI.pDependencies = &dep;

    VkRenderPass renderPass;
    VK_CHECK(vkCreateRenderPass(device, &rpCI, nullptr, &renderPass));
    return renderPass;
}

//=============================================================================
// Descriptor Set Layout (14 inputs + compute output SSBO + color storage image)
//=============================================================================

static VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device)
{
    constexpr uint32_t kNumBindings = 16;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < kNumBindings; i++) {
        VkDescriptorSetLayoutBinding b = {};
        b.binding = i;
        b.descriptorType =
            i == 15 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        if (i >= 14) {
            b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        } else {
            b.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT
                         | VK_SHADER_STAGE_MESH_BIT_EXT
                         | VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        bindings.push_back(b);
    }

    VkDescriptorSetLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = (uint32_t)bindings.size();
    ci.pBindings = bindings.data();

    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout));
    return layout;
}

//=============================================================================
// Pipeline Layout
//=============================================================================

static VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout dsLayout)
{
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = kLudusPushConstantStages;
    pushRange.offset = 0;
    pushRange.size = sizeof(LudusPushConstants);

    VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &dsLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pushRange;

    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(device, &ci, nullptr, &layout));
    return layout;
}

//=============================================================================
// Mesh Pipeline (task + mesh + fragment)
//=============================================================================

static VkPipeline createMeshPipeline(
    VkDevice device,
    VkPipelineLayout layout,
    VkRenderPass renderPass,
    VkShaderModule taskModule,
    VkShaderModule meshModule,
    VkShaderModule fragModule,
    VkSampleCountFlagBits samples)
{
    std::vector<VkPipelineShaderStageCreateInfo> stages;

    if (taskModule != VK_NULL_HANDLE) {
        VkPipelineShaderStageCreateInfo taskStage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        taskStage.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
        taskStage.module = taskModule;
        taskStage.pName = "main";
        stages.push_back(taskStage);
    }

    VkPipelineShaderStageCreateInfo meshStage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    meshStage.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    meshStage.module = meshModule;
    meshStage.pName = "main";
    stages.push_back(meshStage);

    VkPipelineShaderStageCreateInfo fragStage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";
    stages.push_back(fragStage);

    VkPipelineViewportStateCreateInfo viewportState = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterState = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterState.cullMode = VK_CULL_MODE_NONE;
    rasterState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterState.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaaState = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaaState.rasterizationSamples = samples;

    VkPipelineDepthStencilStateCreateInfo depthState = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthState.depthTestEnable = VK_TRUE;
    depthState.depthWriteEnable = VK_TRUE;
    depthState.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment = {};
    blendAttachment.blendEnable = VK_FALSE;
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
                                   | VK_COLOR_COMPONENT_G_BIT
                                   | VK_COLOR_COMPONENT_B_BIT
                                   | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blendState = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blendState.attachmentCount = 1;
    blendState.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineCI = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineCI.stageCount = (uint32_t)stages.size();
    pipelineCI.pStages = stages.data();
    pipelineCI.pViewportState = &viewportState;
    pipelineCI.pRasterizationState = &rasterState;
    pipelineCI.pMultisampleState = &msaaState;
    pipelineCI.pDepthStencilState = &depthState;
    pipelineCI.pColorBlendState = &blendState;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.layout = layout;
    pipelineCI.renderPass = renderPass;
    pipelineCI.subpass = 0;

    VkPipeline pipeline;
    VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline);
    TORCH_CHECK(r == VK_SUCCESS, "vkCreateGraphicsPipelines failed for mesh pipeline: ", (int)r);
    return pipeline;
}

static VkPipeline createComputePipeline(
    VkDevice device, VkPipelineLayout layout, VkShaderModule module)
{
    VkPipelineShaderStageCreateInfo stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";

    VkComputePipelineCreateInfo ci = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = stage;
    ci.layout = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline));
    return pipeline;
}

//=============================================================================
// Descriptor Set Update
//=============================================================================

static void updateDescriptorSet(VkDevice device, VkDescriptorSet& ds, LudusTimestampedVkState& s)
{
    struct BufInfo { uint32_t binding; VkBuffer buffer; VkDeviceSize size; };
    BufInfo bufs[] = {
        { 0, s.timestampsBuffer.buffer,       s.timestampsBuffer.size},
        { 1, s.int32Buffer.buffer,            s.int32Buffer.size},
        { 2, s.vertexBuffer.buffer,           s.vertexBuffer.size},
        { 3, s.triangleBuffer.buffer,         s.triangleBuffer.size},
        { 4, s.poseBuffer.buffer,             s.poseBuffer.size},
        { 5, s.floatBuffer.buffer,            s.floatBuffer.size},
        { 6, s.sceneBuffer.buffer,            s.sceneBuffer.size},
        { 7, s.polylinePoolBuffer.buffer,     s.polylinePoolBuffer.size},
        { 8, s.polygonPoolBuffer.buffer,      s.polygonPoolBuffer.size},
        { 9, s.obstaclePoolBuffer.buffer,     s.obstaclePoolBuffer.size},
        {10, s.colorPaletteBuffer.buffer,     s.colorPaletteBuffer.size},
        {11, s.cameraIntrinsicsBuffer.buffer, s.cameraIntrinsicsBuffer.size},
        {12, s.cameraPoseBuffer.buffer,       s.cameraPoseBuffer.size},
        {13, s.queryBuffer.buffer,            s.queryBuffer.size},
    };

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> bufInfos(14);

    for (int i = 0; i < 14; i++) {
        if (bufs[i].buffer == VK_NULL_HANDLE || bufs[i].size == 0)
            continue;

        bufInfos[i].buffer = bufs[i].buffer;
        bufInfos[i].offset = 0;
        bufInfos[i].range = bufs[i].size;

        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = ds;
        w.dstBinding = bufs[i].binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bufInfos[i];
        writes.push_back(w);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    }
}

static void updateAllInputDescriptorSets(LudusTimestampedVkState& s)
{
    for (uint32_t i = 0; i < VkContext::kFramesInFlight; ++i)
        updateDescriptorSet(s.vkctx.device, s.descriptorSets[i], s);
}

static void updateFrameExportDescriptors(
    VkDevice device, VkDescriptorSet ds, const VkExternalBuffer& linearOutput,
    VkDeviceSize bytes, const VkDeviceImage& colorAttachment)
{
    VkDescriptorBufferInfo info = {};
    info.buffer = linearOutput.buffer;
    info.offset = 0;
    info.range = bytes;
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageView = colorAttachment.imageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ds;
    writes[0].dstBinding = 14;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &info;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = ds;
    writes[1].dstBinding = 15;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

//=============================================================================
// Initialization
//=============================================================================

void ludusTimestampedInitVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s, int cudaDeviceIdx)
{
    (void)nvdr_ctx;
    memset(&s, 0, sizeof(s));

    // cuImportExternalMemory and friends require a current CUDA context.
    // Force the runtime to materialize a primary context on the requested
    // device before any external-memory imports happen during init.
    cudaSetDevice(cudaDeviceIdx);
    cudaFree(nullptr);

    s.vkctx = createVkContext(cudaDeviceIdx);
    s.hasMeshShader = s.vkctx.hasMeshShader ? 1 : 0;

    s.msaaSamples = 0;
    s.tessellationThreshold = 1.0f;
    s.maxTessellationLevelPolyline = 4;
    s.maxTessellationLevelPolygon = 3;
    s.maxTessellationLevelCube = 3;
    s.depthScaling = 1.0f;
    s.maxExtrapolationUs = 500000;
    s.cullRadiusScale = 1.5f;

    s.renderPass = createTimestampedRenderPass(
        s.vkctx.device, VK_FORMAT_D24_UNORM_S8_UINT, VK_SAMPLE_COUNT_1_BIT
    );

    s.descriptorSetLayout = createDescriptorSetLayout(s.vkctx.device);
    s.pipelineLayout = createPipelineLayout(s.vkctx.device, s.descriptorSetLayout);

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 15 * VkContext::kFramesInFlight;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = VkContext::kFramesInFlight;

    VkDescriptorPoolCreateInfo poolCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets = VkContext::kFramesInFlight;
    poolCI.poolSizeCount = 2;
    poolCI.pPoolSizes = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(s.vkctx.device, &poolCI, nullptr, &s.descriptorPool));

    VkDescriptorSetAllocateInfo dsAllocCI = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAllocCI.descriptorPool = s.descriptorPool;
    VkDescriptorSetLayout layouts[VkContext::kFramesInFlight];
    for (uint32_t i = 0; i < VkContext::kFramesInFlight; ++i)
        layouts[i] = s.descriptorSetLayout;
    dsAllocCI.descriptorSetCount = VkContext::kFramesInFlight;
    dsAllocCI.pSetLayouts = layouts;
    VK_CHECK(vkAllocateDescriptorSets(
        s.vkctx.device, &dsAllocCI, s.descriptorSets));

    // Embedded SPIR-V (generated from shaders/*.spv at build time).
    struct ShaderBin { const uint32_t* code; size_t bytes; };
    const ShaderBin bins[10] = {
        {kSpv_ts_polyline_task, sizeof(kSpv_ts_polyline_task)},
        {kSpv_ts_polyline_mesh, sizeof(kSpv_ts_polyline_mesh)},
        {kSpv_ts_polyline_frag, sizeof(kSpv_ts_polyline_frag)},
        {kSpv_ts_polygon_task,  sizeof(kSpv_ts_polygon_task)},
        {kSpv_ts_polygon_mesh,  sizeof(kSpv_ts_polygon_mesh)},
        {kSpv_ts_polygon_frag,  sizeof(kSpv_ts_polygon_frag)},
        {kSpv_ts_obstacle_task, sizeof(kSpv_ts_obstacle_task)},
        {kSpv_ts_obstacle_mesh, sizeof(kSpv_ts_obstacle_mesh)},
        {kSpv_ts_obstacle_frag, sizeof(kSpv_ts_obstacle_frag)},
        {kSpv_ts_export_comp,   sizeof(kSpv_ts_export_comp)},
    };
    for (int i = 0; i < 10; i++)
        s.shaderModules[i] = createShaderModule(s.vkctx.device, bins[i].code, bins[i].bytes);

    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    s.pipelinePolyline = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[0], s.shaderModules[1], s.shaderModules[2], samples);
    s.pipelinePolygon  = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[3], s.shaderModules[4], s.shaderModules[5], samples);
    s.pipelineObstacle = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[6], s.shaderModules[7], s.shaderModules[8], samples);
    s.pipelineExport = createComputePipeline(
        s.vkctx.device, s.pipelineLayout, s.shaderModules[9]);
    s.renderPassSamples = 1;  // render pass + pipelines built single-sample above

    // Dummy buffers so every descriptor binding has a valid buffer before
    // the first upload_scene call. Placeholders are deliberately not exported;
    // the first real upload grows/upgrades only the required buffers and then
    // performs their one-time Vulkan->external ownership release.
    VkDeviceSize dummySize = 256;
    VkBufferUsageFlags ssboUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    auto makeDummy = [&](VkExternalBuffer& buf, bool needsCuda = false) {
        if (buf.buffer == VK_NULL_HANDLE)
            buf = createExternalBuffer(s.vkctx, dummySize, ssboUsage, needsCuda);
    };
    makeDummy(s.timestampsBuffer);
    makeDummy(s.int32Buffer);
    makeDummy(s.vertexBuffer);
    makeDummy(s.triangleBuffer);
    makeDummy(s.poseBuffer);
    makeDummy(s.floatBuffer);
    // Scene buffer: smaller dummy so uploadScene always creates a larger one
    s.sceneBuffer = createExternalBuffer(s.vkctx, 64, ssboUsage, false);
    makeDummy(s.polylinePoolBuffer);
    makeDummy(s.polygonPoolBuffer);
    makeDummy(s.obstaclePoolBuffer);
    makeDummy(s.colorPaletteBuffer);
    makeDummy(s.cameraIntrinsicsBuffer);
    makeDummy(s.cameraPoseBuffer);
    makeDummy(s.queryBuffer);
    updateAllInputDescriptorSets(s);

    cudaStreamCreate(&s.copyStream);
    for (int i = 0; i < 2; i++) {
        cudaEventCreateWithFlags(&s.stagingReadyEvent[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&s.pinnedReadyEvent[i], cudaEventDisableTiming);
    }

    VK_DBG("[Vulkan] Timestamped renderer initialized\n");
}

//=============================================================================
// Buffer Resize Helpers
//=============================================================================

static const VkBufferUsageFlags SSBO_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

static void synchronizeForResourceMutation(
    LudusTimestampedVkState& s, cudaStream_t stream)
{
    // Buffer/image handle replacement is a control-plane operation. Drain the
    // last shared-memory consumer on this stream, then wait for Vulkan before
    // destroying handles. This path runs only on capacity/resolution changes;
    // steady-state rendering remains fully GPU ordered.
    waitInteropTimelineOnCuda(s.vkctx, stream, s.lastVulkanDoneValue);
    cudaError_t syncError = cudaStreamSynchronize(stream);
    TORCH_CHECK(syncError == cudaSuccess,
        "cudaStreamSynchronize during Vulkan resource mutation failed: ",
        (int)syncError);
    VK_CHECK(vkDeviceWaitIdle(s.vkctx.device));
}

static bool externalBufferNeedsResize(
    const VkExternalBuffer& buffer, VkDeviceSize size)
{
    return size > 0 &&
        (buffer.buffer == VK_NULL_HANDLE || buffer.size < size || buffer.cuDevPtr == 0);
}

static void releaseNewExternalBuffersToCuda(
    LudusTimestampedVkState& s,
    const std::vector<VkExternalBuffer*>& buffers)
{
    if (buffers.empty()) return;
    VkCommandBuffer cmd = beginSingleTimeCommands(s.vkctx);
    std::vector<VkBufferMemoryBarrier> barriers;
    for (VkExternalBuffer* buffer : buffers) {
        if (!buffer || buffer->buffer == VK_NULL_HANDLE || !buffer->cuExtMem)
            continue;
        VkBufferMemoryBarrier release = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        release.srcAccessMask = 0;
        release.dstAccessMask = 0;
        release.srcQueueFamilyIndex = s.vkctx.graphicsQueueFamily;
        release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        release.buffer = buffer->buffer;
        release.offset = 0;
        release.size = VK_WHOLE_SIZE;
        barriers.push_back(release);
    }
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, (uint32_t)barriers.size(), barriers.data(), 0, nullptr);
    endSingleTimeCommands(s.vkctx, cmd);
}

struct PendingSharedBufferGrowth
{
    VkExternalBuffer* slot;
    VkExternalBuffer previous;
    VkDeviceSize preserveBytes;
};

static int growCapacityGeometrically(int capacity, int needed)
{
    if (needed <= capacity) return capacity;
    int grown = std::max(1, capacity);
    while (grown < needed && grown <= INT_MAX / 2) grown *= 2;
    return std::max(grown, needed);
}

static void queueSharedBufferGrowth(
    LudusTimestampedVkState& s,
    VkExternalBuffer& buffer,
    VkDeviceSize targetBytes,
    VkDeviceSize preserveBytes,
    std::vector<PendingSharedBufferGrowth>& pending)
{
    if (targetBytes == 0) return;
    const bool needsCudaUpgrade = buffer.cuDevPtr == 0;
    const bool needsGrowth =
        buffer.buffer == VK_NULL_HANDLE || buffer.size < targetBytes;
    if (!needsCudaUpgrade && !needsGrowth) return;

    PendingSharedBufferGrowth growth = {};
    growth.slot = &buffer;
    growth.previous = buffer;
    growth.preserveBytes = preserveBytes;
    buffer = createExternalBuffer(
        s.vkctx, std::max(targetBytes, growth.previous.size), SSBO_USAGE, true);
    pending.push_back(growth);
}

static void finishSharedBufferGrowth(
    LudusTimestampedVkState& s,
    cudaStream_t stream,
    std::vector<PendingSharedBufferGrowth>& pending)
{
    if (pending.empty()) return;

    std::vector<VkExternalBuffer*> replacements;
    replacements.reserve(pending.size());
    for (PendingSharedBufferGrowth& growth : pending)
        replacements.push_back(growth.slot);
    releaseNewExternalBuffersToCuda(s, replacements);

    bool copied = false;
    for (PendingSharedBufferGrowth& growth : pending) {
        if (growth.preserveBytes == 0) continue;
        TORCH_CHECK(growth.previous.cuDevPtr != 0,
            "cannot preserve a Vulkan shared buffer without an old CUDA mapping");
        TORCH_CHECK(growth.preserveBytes <= growth.previous.size,
            "preserved Vulkan buffer range exceeds the old allocation");
        cudaError_t error = cudaMemcpyAsync(
            (void*)growth.slot->cuDevPtr,
            (const void*)growth.previous.cuDevPtr,
            (size_t)growth.preserveBytes,
            cudaMemcpyDeviceToDevice,
            stream);
        TORCH_CHECK(error == cudaSuccess,
            "cudaMemcpyAsync while preserving Vulkan buffer growth failed: ",
            (int)error);
        copied = true;
    }

    // Growth is a rare control-plane operation. The old external allocations
    // cannot be destroyed until their stream-ordered preservation copies end.
    if (copied) {
        cudaError_t error = cudaStreamSynchronize(stream);
        TORCH_CHECK(error == cudaSuccess,
            "cudaStreamSynchronize after preserving Vulkan buffers failed: ",
            (int)error);
    }
    for (PendingSharedBufferGrowth& growth : pending)
        destroyExternalBuffer(s.vkctx, growth.previous);
}

// Rebuild the render pass + pipelines for a new sample count (their attachment
// layout and rasterizationSamples are fixed at creation). Caller must be idle.
static void rebuildRenderPassAndPipelines(LudusTimestampedVkState& s, VkSampleCountFlagBits samples)
{
    if (s.pipelinePolyline) { vkDestroyPipeline(s.vkctx.device, s.pipelinePolyline, nullptr); s.pipelinePolyline = VK_NULL_HANDLE; }
    if (s.pipelinePolygon)  { vkDestroyPipeline(s.vkctx.device, s.pipelinePolygon,  nullptr); s.pipelinePolygon  = VK_NULL_HANDLE; }
    if (s.pipelineObstacle) { vkDestroyPipeline(s.vkctx.device, s.pipelineObstacle, nullptr); s.pipelineObstacle = VK_NULL_HANDLE; }
    if (s.renderPass)       { vkDestroyRenderPass(s.vkctx.device, s.renderPass, nullptr); s.renderPass = VK_NULL_HANDLE; }

    s.renderPass = createTimestampedRenderPass(
        s.vkctx.device, VK_FORMAT_D24_UNORM_S8_UINT, samples);

    s.pipelinePolyline = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[0], s.shaderModules[1], s.shaderModules[2], samples);
    s.pipelinePolygon  = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[3], s.shaderModules[4], s.shaderModules[5], samples);
    s.pipelineObstacle = createMeshPipeline(s.vkctx.device, s.pipelineLayout, s.renderPass,
        s.shaderModules[6], s.shaderModules[7], s.shaderModules[8], samples);

    s.renderPassSamples = (samples == VK_SAMPLE_COUNT_1_BIT) ? 1 : (int)samples;
}

static void destroyRenderTarget(
    LudusTimestampedVkState& s, LudusVkRenderTarget& target)
{
    if (target.framebuffer)
        vkDestroyFramebuffer(s.vkctx.device, target.framebuffer, nullptr);
    destroyDeviceImage(s.vkctx, target.colorAttachment);
    destroyDeviceImage(s.vkctx, target.depthStencilAttachment);
    memset(&target, 0, sizeof(target));
}

static void destroyRenderTargets(LudusTimestampedVkState& s)
{
    for (int i = 0; i < s.numRenderTargets; ++i)
        destroyRenderTarget(s, s.renderTargets[i]);
    s.numRenderTargets = 0;
    s.activeRenderTarget = -1;
}

static void ensureFramebuffer(
    LudusTimestampedVkState& s,
    cudaStream_t stream,
    int width,
    int height,
    int layers)
{
    int desiredSamplesInt = (s.msaaSamples > 1) ? s.msaaSamples : 1;
    for (int i = 0; i < s.numRenderTargets; ++i) {
        const LudusVkRenderTarget& target = s.renderTargets[i];
        if (target.width == width && target.height == height &&
            target.samples == desiredSamplesInt && target.maxLayers >= layers) {
            s.activeRenderTarget = i;
            return;
        }
    }

    synchronizeForResourceMutation(s, stream);
    if (s.renderPassSamples != desiredSamplesInt) {
        destroyRenderTargets(s);
        rebuildRenderPassAndPipelines(
            s, (VkSampleCountFlagBits)desiredSamplesInt);
    }

    int targetIndex = -1;
    for (int i = 0; i < s.numRenderTargets; ++i) {
        if (s.renderTargets[i].width == width &&
            s.renderTargets[i].height == height) {
            targetIndex = i;
            destroyRenderTarget(s, s.renderTargets[i]);
            break;
        }
    }
    if (targetIndex < 0) {
        if (s.numRenderTargets < LUDUS_VK_RENDER_TARGET_CACHE_SIZE) {
            targetIndex = s.numRenderTargets++;
        } else {
            targetIndex = 0;
            destroyRenderTarget(s, s.renderTargets[targetIndex]);
        }
    }

    LudusVkRenderTarget& target = s.renderTargets[targetIndex];
    target.width = width;
    target.height = height;
    target.maxLayers = layers;
    target.samples = desiredSamplesInt;
    target.colorAttachment = createDeviceImage(s.vkctx, width, height, layers,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        (VkSampleCountFlagBits)desiredSamplesInt);
    target.depthStencilAttachment = createDeviceImage(s.vkctx, width, height, layers,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        (VkSampleCountFlagBits)desiredSamplesInt);

    VkFramebufferCreateInfo fbCI = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbCI.renderPass = s.renderPass;
    VkImageView attachments[] = {
        target.colorAttachment.imageView,
        target.depthStencilAttachment.imageView,
    };
    fbCI.attachmentCount = 2;
    fbCI.pAttachments = attachments;
    fbCI.width = width;
    fbCI.height = height;
    fbCI.layers = layers;
    VK_CHECK(vkCreateFramebuffer(
        s.vkctx.device, &fbCI, nullptr, &target.framebuffer));
    s.activeRenderTarget = targetIndex;

    // CUDA staging is optional and allocated lazily only by the legacy
    // staging/JPEG API. Interactive drive reuses this cached color/depth target.
}

//=============================================================================
// Scene Upload
//=============================================================================

void ludusUploadCamerasVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const FThetaCamera* intrinsics, int numCameras)
{
    (void)nvdr_ctx;
    if (numCameras > s.cameraCapacity) {
        synchronizeForResourceMutation(s, stream);
        s.cameraCapacity = numCameras;
        resizeExternalBuffer(s.vkctx, s.cameraIntrinsicsBuffer,
            numCameras * sizeof(FThetaCamera), SSBO_USAGE, true);
        releaseNewExternalBuffersToCuda(s, {&s.cameraIntrinsicsBuffer});
        updateAllInputDescriptorSets(s);
    } else {
        waitInteropTimelineOnCuda(s.vkctx, stream, s.lastVulkanDoneValue);
    }
    s.numCameras = numCameras;

    cudaMemcpyAsync((void*)s.cameraIntrinsicsBuffer.cuDevPtr, intrinsics,
        numCameras * sizeof(FThetaCamera), cudaMemcpyDeviceToDevice, stream);
    s.sceneBuffersDirty = 1;
}

void ludusUploadColorPaletteVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const float* colors, int numColors)
{
    (void)nvdr_ctx;
    VkDeviceSize size = numColors * 4 * sizeof(float);
    if (externalBufferNeedsResize(s.colorPaletteBuffer, size)) {
        synchronizeForResourceMutation(s, stream);
        resizeExternalBuffer(s.vkctx, s.colorPaletteBuffer, size, SSBO_USAGE, true);
        releaseNewExternalBuffersToCuda(s, {&s.colorPaletteBuffer});
        updateAllInputDescriptorSets(s);
    } else {
        waitInteropTimelineOnCuda(s.vkctx, stream, s.lastVulkanDoneValue);
    }
    cudaMemcpyAsync((void*)s.colorPaletteBuffer.cuDevPtr, colors, size,
        cudaMemcpyDeviceToDevice, stream);
    s.colorPaletteSize = numColors;
    s.sceneBuffersDirty = 1;
}

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
    const float* floatData, int numFloats)
{
    (void)nvdr_ctx;
    int sceneId = s.numScenes++;

    s.timestampsCapacity = growCapacityGeometrically(
        s.timestampsCapacity, s.timestampsUsed + numTimestamps);
    s.int32Capacity = growCapacityGeometrically(
        s.int32Capacity, s.int32Used + numInt32);
    s.vertexCapacity = growCapacityGeometrically(
        s.vertexCapacity, s.vertexUsed + numVertices);
    s.triangleCapacity = growCapacityGeometrically(
        s.triangleCapacity, s.triangleUsed + numTriangles);
    s.poseCapacity = growCapacityGeometrically(
        s.poseCapacity, s.poseUsed + numPoses);
    s.floatCapacity = growCapacityGeometrically(
        s.floatCapacity, s.floatUsed + numFloats);
    s.polylinePoolCapacity = growCapacityGeometrically(
        s.polylinePoolCapacity, s.polylinePoolUsed + numPolylinePools);
    s.polygonPoolCapacity = growCapacityGeometrically(
        s.polygonPoolCapacity, s.polygonPoolUsed + numPolygonPools);
    s.obstaclePoolCapacity = growCapacityGeometrically(
        s.obstaclePoolCapacity, s.obstaclePoolUsed + numObstaclePools);
    s.maxObstaclesPerPool       = std::max(s.maxObstaclesPerPool,       maxObstaclesInPool);
    s.maxPolylinePoolsPerScene  = std::max(s.maxPolylinePoolsPerScene,  numPolylinePools);
    s.maxPolygonPoolsPerScene   = std::max(s.maxPolygonPoolsPerScene,   numPolygonPools);
    s.maxCubePoolsPerScene      = std::max(s.maxCubePoolsPerScene,      numObstaclePools);
    s.maxVarraysPerTsPolyline   = std::max(s.maxVarraysPerTsPolyline,   maxVarraysPerTsPolyline);
    s.maxVarraysPerTsPolygon    = std::max(s.maxVarraysPerTsPolygon,    maxVarraysPerTsPolygon);

    int sceneCapNeeded = sceneId + 1;
    const bool sceneBufferNeedsResize = sceneCapNeeded > s.maxScenes;
    const bool dataBuffersNeedResize =
        externalBufferNeedsResize(s.timestampsBuffer,
            (VkDeviceSize)s.timestampsCapacity * sizeof(int64_t)) ||
        externalBufferNeedsResize(s.int32Buffer,
            (VkDeviceSize)s.int32Capacity * sizeof(int32_t)) ||
        externalBufferNeedsResize(s.vertexBuffer,
            (VkDeviceSize)s.vertexCapacity * sizeof(Vertex)) ||
        externalBufferNeedsResize(s.triangleBuffer,
            (VkDeviceSize)s.triangleCapacity * sizeof(Triangle)) ||
        externalBufferNeedsResize(s.poseBuffer,
            (VkDeviceSize)s.poseCapacity * sizeof(CameraPose)) ||
        externalBufferNeedsResize(s.floatBuffer,
            (VkDeviceSize)s.floatCapacity * sizeof(float)) ||
        externalBufferNeedsResize(s.polylinePoolBuffer,
            (VkDeviceSize)s.polylinePoolCapacity * sizeof(TimestampedPolylinePool)) ||
        externalBufferNeedsResize(s.polygonPoolBuffer,
            (VkDeviceSize)s.polygonPoolCapacity * sizeof(TimestampedPolygonPool)) ||
        externalBufferNeedsResize(s.obstaclePoolBuffer,
            (VkDeviceSize)s.obstaclePoolCapacity * sizeof(ObstaclePool));
    if (sceneBufferNeedsResize || dataBuffersNeedResize) {
        synchronizeForResourceMutation(s, stream);
    } else {
        waitInteropTimelineOnCuda(s.vkctx, stream, s.lastVulkanDoneValue);
    }

    std::vector<PendingSharedBufferGrowth> pendingGrowth;
    pendingGrowth.reserve(10);
    if (sceneBufferNeedsResize) {
        s.maxScenes = growCapacityGeometrically(s.maxScenes, sceneCapNeeded);
        queueSharedBufferGrowth(
            s, s.sceneBuffer,
            (VkDeviceSize)s.maxScenes * sizeof(TimestampedScene),
            (VkDeviceSize)sceneId * sizeof(TimestampedScene),
            pendingGrowth);
    }
    queueSharedBufferGrowth(s, s.timestampsBuffer,
        (VkDeviceSize)s.timestampsCapacity * sizeof(int64_t),
        (VkDeviceSize)s.timestampsUsed * sizeof(int64_t), pendingGrowth);
    queueSharedBufferGrowth(s, s.int32Buffer,
        (VkDeviceSize)s.int32Capacity * sizeof(int32_t),
        (VkDeviceSize)s.int32Used * sizeof(int32_t), pendingGrowth);
    queueSharedBufferGrowth(s, s.vertexBuffer,
        (VkDeviceSize)s.vertexCapacity * sizeof(Vertex),
        (VkDeviceSize)s.vertexUsed * sizeof(Vertex), pendingGrowth);
    queueSharedBufferGrowth(s, s.triangleBuffer,
        (VkDeviceSize)s.triangleCapacity * sizeof(Triangle),
        (VkDeviceSize)s.triangleUsed * sizeof(Triangle), pendingGrowth);
    queueSharedBufferGrowth(s, s.poseBuffer,
        (VkDeviceSize)s.poseCapacity * sizeof(CameraPose),
        (VkDeviceSize)s.poseUsed * sizeof(CameraPose), pendingGrowth);
    queueSharedBufferGrowth(s, s.floatBuffer,
        (VkDeviceSize)s.floatCapacity * sizeof(float),
        (VkDeviceSize)s.floatUsed * sizeof(float), pendingGrowth);
    queueSharedBufferGrowth(s, s.polylinePoolBuffer,
        (VkDeviceSize)s.polylinePoolCapacity * sizeof(TimestampedPolylinePool),
        (VkDeviceSize)s.polylinePoolUsed * sizeof(TimestampedPolylinePool),
        pendingGrowth);
    queueSharedBufferGrowth(s, s.polygonPoolBuffer,
        (VkDeviceSize)s.polygonPoolCapacity * sizeof(TimestampedPolygonPool),
        (VkDeviceSize)s.polygonPoolUsed * sizeof(TimestampedPolygonPool),
        pendingGrowth);
    queueSharedBufferGrowth(s, s.obstaclePoolBuffer,
        (VkDeviceSize)s.obstaclePoolCapacity * sizeof(ObstaclePool),
        (VkDeviceSize)s.obstaclePoolUsed * sizeof(ObstaclePool), pendingGrowth);
    if (!pendingGrowth.empty()) {
        finishSharedBufferGrowth(s, stream, pendingGrowth);
        updateAllInputDescriptorSets(s);
    }

    auto copyAppend = [&](VkExternalBuffer& buf, const void* data, int count, size_t elemSize,
                          int& used, const char* tag) {
        if (count > 0 && data) {
            TORCH_CHECK(buf.cuDevPtr != 0,
                "upload_scene: buffer ", tag, " has no CUDA pointer (cudaImportable not set?)");
            size_t dstOff = (size_t)used * elemSize;
            size_t bytes = (size_t)count * elemSize;
            CUdeviceptr dst = buf.cuDevPtr + dstOff;
            cudaError_t e = cudaMemcpyAsync((void*)dst, data, bytes,
                cudaMemcpyDeviceToDevice, stream);
            TORCH_CHECK(e == cudaSuccess, "upload_scene: cudaMemcpyAsync ", tag,
                " failed: ", (int)e);
            VK_DBG("[Vulkan] upload %6s: count=%d bytes=%zu\n", tag, count, bytes);
        }
        int offset = used;
        used += count;
        return offset;
    };

    copyAppend(s.timestampsBuffer,   timestamps,    numTimestamps,    sizeof(int64_t),                  s.timestampsUsed, "ts");
    copyAppend(s.int32Buffer,        int32Data,     numInt32,         sizeof(int32_t),                  s.int32Used,      "i32");
    copyAppend(s.vertexBuffer,       vertices,      numVertices,      sizeof(Vertex),                   s.vertexUsed,     "vert");
    copyAppend(s.triangleBuffer,     triangles,     numTriangles,     sizeof(Triangle),                 s.triangleUsed,   "tri");
    copyAppend(s.poseBuffer,         poses,         numPoses,         sizeof(CameraPose),               s.poseUsed,       "pose");
    copyAppend(s.floatBuffer,        floatData,     numFloats,        sizeof(float),                    s.floatUsed,      "flt");
    copyAppend(s.polylinePoolBuffer, polylinePools, numPolylinePools, sizeof(TimestampedPolylinePool),  s.polylinePoolUsed, "pl");
    copyAppend(s.polygonPoolBuffer,  polygonPools,  numPolygonPools,  sizeof(TimestampedPolygonPool),   s.polygonPoolUsed,  "pg");
    copyAppend(s.obstaclePoolBuffer, obstaclePools, numObstaclePools, sizeof(ObstaclePool),             s.obstaclePoolUsed, "obs");

    TORCH_CHECK(s.sceneBuffer.cuDevPtr != 0,
        "upload_scene: sceneBuffer has no CUDA pointer (cudaImportable not set?)");
    cudaError_t e_scene = cudaMemcpyAsync(
        (void*)(s.sceneBuffer.cuDevPtr + sceneId * sizeof(TimestampedScene)),
        sceneDesc, sizeof(TimestampedScene), cudaMemcpyDeviceToDevice, stream);
    TORCH_CHECK(e_scene == cudaSuccess,
        "upload_scene: cudaMemcpyAsync (scene desc) failed: ", (int)e_scene);

    s.sceneBuffersDirty = 1;
    return sceneId;
}

void ludusUpdateCubePoolVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    int absolutePoolIndex,
    int timestampOffset, int trackTimestampOffset, int int32Offset,
    int translationOffset, int quaternionOffset, int scaleOffset, int colorOffset,
    const int64_t* timestamps, int numTimestamps,
    const int64_t* trackTimestamps, int numTrackTimestamps,
    const int32_t* prefixSum, int numCubes,
    const float* translations, int numTranslationFloats,
    const float* quaternions, int numQuaternionFloats,
    const float* scales, int numScaleFloats,
    const float* colors, int numColorFloats,
    const ObstaclePool* poolHeader)
{
    (void)nvdr_ctx;
    waitInteropTimelineOnCuda(s.vkctx, stream, s.lastVulkanDoneValue);
    auto copyAt = [&](VkExternalBuffer& buffer, size_t elementOffset,
                      const void* source, size_t bytes, const char* tag) {
        if (bytes == 0) return;
        TORCH_CHECK(buffer.cuDevPtr != 0,
            "update_cube_pool: shared buffer ", tag, " has no CUDA mapping");
        cudaError_t error = cudaMemcpyAsync(
            (void*)(buffer.cuDevPtr + elementOffset), source, bytes,
            cudaMemcpyDeviceToDevice, stream);
        TORCH_CHECK(error == cudaSuccess,
            "update_cube_pool: cudaMemcpyAsync ", tag, " failed: ", (int)error);
    };

    copyAt(s.timestampsBuffer, (size_t)timestampOffset * sizeof(int64_t),
        timestamps, (size_t)numTimestamps * sizeof(int64_t), "timestamps");
    copyAt(s.timestampsBuffer, (size_t)trackTimestampOffset * sizeof(int64_t),
        trackTimestamps, (size_t)numTrackTimestamps * sizeof(int64_t),
        "track timestamps");
    copyAt(s.int32Buffer, (size_t)int32Offset * sizeof(int32_t),
        prefixSum, (size_t)numCubes * sizeof(int32_t), "prefix sum");
    copyAt(s.floatBuffer, (size_t)translationOffset * sizeof(float),
        translations, (size_t)numTranslationFloats * sizeof(float), "translations");
    copyAt(s.floatBuffer, (size_t)quaternionOffset * sizeof(float),
        quaternions, (size_t)numQuaternionFloats * sizeof(float), "quaternions");
    copyAt(s.floatBuffer, (size_t)scaleOffset * sizeof(float),
        scales, (size_t)numScaleFloats * sizeof(float), "scales");
    copyAt(s.floatBuffer, (size_t)colorOffset * sizeof(float),
        colors, (size_t)numColorFloats * sizeof(float), "colors");
    copyAt(s.obstaclePoolBuffer,
        (size_t)absolutePoolIndex * sizeof(ObstaclePool),
        poolHeader, sizeof(ObstaclePool), "pool header");
}

void ludusRemoveSceneVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s, int sceneId, cudaStream_t stream)
{
    (void)nvdr_ctx;
    // Tombstone with a stream-ordered memset; no host lifetime or CPU wait.
    waitInteropTimelineOnCuda(s.vkctx, stream, s.lastVulkanDoneValue);
    cudaError_t e = cudaMemsetAsync(
        (void*)(s.sceneBuffer.cuDevPtr + sceneId * sizeof(TimestampedScene)),
        0, sizeof(int), stream);
    TORCH_CHECK(e == cudaSuccess, "remove_scene: cudaMemsetAsync (tombstone) failed: ", (int)e);
    s.sceneBuffersDirty = 1;
}

//=============================================================================
// Render Batch
//=============================================================================

void ludusRenderBatchVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    const RenderQuery* queries, const CameraPose* cameraPoses,
    int numQueries, int width, int height, int outputSlot)
{
    (void)nvdr_ctx;
    if (numQueries <= 0) return;
    TORCH_CHECK(outputSlot >= 0 && outputSlot < LUDUS_VK_OUTPUT_SLOTS,
        "invalid Vulkan output slot ", outputSlot);

    VK_DBG("[Vulkan] renderBatch: nq=%d size=%dx%d\n", numQueries, width, height);

    ensureFramebuffer(s, stream, width, height, numQueries);

    if (numQueries > s.queryCapacity) {
        synchronizeForResourceMutation(s, stream);
        s.queryCapacity = numQueries;
        resizeExternalBuffer(s.vkctx, s.queryBuffer,
            numQueries * sizeof(RenderQuery), SSBO_USAGE, true);
        resizeExternalBuffer(s.vkctx, s.cameraPoseBuffer,
            numQueries * sizeof(CameraPose), SSBO_USAGE, true);
        releaseNewExternalBuffersToCuda(
            s, {&s.queryBuffer, &s.cameraPoseBuffer});
        updateAllInputDescriptorSets(s);
    }

    const size_t totalSize = (size_t)width * height * 4 * numQueries;
    LudusVkExportSlot& exportSlot = s.exportSlots[outputSlot];
    VkExternalBuffer& outputBuffer = exportSlot.linearRgba;
    if (externalBufferNeedsResize(outputBuffer, totalSize)) {
        synchronizeForResourceMutation(s, stream);
        resizeExternalBuffer(s.vkctx, outputBuffer, totalSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            true);
        // Fresh allocations start in Vulkan ownership. Reused output slots
        // were released to CUDA by their previous render submission.
        exportSlot.releasedToCuda = 0;
    }
    exportSlot.width = width;
    exportSlot.height = height;
    exportSlot.layers = numQueries;

    // A single stream-ordered timeline signal publishes CUDA-written inputs
    // and releases a reused output lease to Vulkan. No host wait is involved.
    cudaError_t e1 = cudaMemcpyAsync((void*)s.queryBuffer.cuDevPtr, queries,
        numQueries * sizeof(RenderQuery), cudaMemcpyDeviceToDevice, stream);
    cudaError_t e2 = cudaMemcpyAsync((void*)s.cameraPoseBuffer.cuDevPtr, cameraPoses,
        numQueries * sizeof(CameraPose), cudaMemcpyDeviceToDevice, stream);
    TORCH_CHECK(e1 == cudaSuccess, "renderBatch: cudaMemcpyAsync (queries) failed: ", (int)e1);
    TORCH_CHECK(e2 == cudaSuccess, "renderBatch: cudaMemcpyAsync (poses) failed: ", (int)e2);

    const uint64_t cudaReadyValue =
        signalInteropTimelineFromCuda(s.vkctx, stream);

    // Rotate command buffers. A host fence wait occurs only if the CPU gets
    // three submissions ahead; there is no steady-state per-frame fence wait.
    const uint32_t frameSlot =
        s.vkctx.frameCursor++ % VkContext::kFramesInFlight;
    VkFence frameFence = s.vkctx.fences[frameSlot];
    VkResult fenceStatus = vkGetFenceStatus(s.vkctx.device, frameFence);
    if (fenceStatus == VK_NOT_READY) {
        VK_CHECK(vkWaitForFences(
            s.vkctx.device, 1, &frameFence, VK_TRUE, UINT64_MAX));
    } else {
        TORCH_CHECK(fenceStatus == VK_SUCCESS,
            "vkGetFenceStatus failed with VkResult ", (int)fenceStatus);
    }
    VK_CHECK(vkResetFences(s.vkctx.device, 1, &frameFence));
    VkDescriptorSet descriptorSet = s.descriptorSets[frameSlot];
    // This frame slot's fence guarantees its descriptor set is no longer in
    // use. Point compute bindings at the leased output and cached color image
    // without touching the other in-flight sets.
    LudusVkRenderTarget& renderTarget =
        s.renderTargets[s.activeRenderTarget];
    updateFrameExportDescriptors(
        s.vkctx.device, descriptorSet, outputBuffer, totalSize,
        renderTarget.colorAttachment);
    VkCommandBuffer cmd = s.vkctx.commandBuffers[frameSlot];
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo beginCI = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginCI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginCI));

    // Queue family ownership acquire from VK_QUEUE_FAMILY_EXTERNAL.
    // CUDA writes happen on an external queue family; this barrier transfers
    // ownership to the graphics queue and makes the writes visible to shaders.
    {
        std::vector<VkBufferMemoryBarrier> bufBarriers;
        auto addBarrier = [&](VkExternalBuffer& buf, VkAccessFlags dstAccess) {
            if (buf.buffer == VK_NULL_HANDLE || buf.size == 0 || !buf.cuExtMem) return;
            VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            b.srcAccessMask = 0;
            b.dstAccessMask = dstAccess;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            b.dstQueueFamilyIndex = s.vkctx.graphicsQueueFamily;
            b.buffer = buf.buffer;
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            bufBarriers.push_back(b);
        };
        addBarrier(s.timestampsBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.int32Buffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.vertexBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.triangleBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.poseBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.floatBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.sceneBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.polylinePoolBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.polygonPoolBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.obstaclePoolBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.colorPaletteBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.cameraIntrinsicsBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.cameraPoseBuffer, VK_ACCESS_SHADER_READ_BIT);
        addBarrier(s.queryBuffer, VK_ACCESS_SHADER_READ_BIT);
        if (exportSlot.releasedToCuda) {
            addBarrier(outputBuffer, VK_ACCESS_SHADER_WRITE_BIT);
        } else {
            VkBufferMemoryBarrier initialOutput = {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            initialOutput.srcAccessMask = 0;
            initialOutput.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            initialOutput.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            initialOutput.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            initialOutput.buffer = outputBuffer.buffer;
            initialOutput.offset = 0;
            initialOutput.size = VK_WHOLE_SIZE;
            bufBarriers.push_back(initialOutput);
        }

        if (!bufBarriers.empty()) {
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT
                  | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT
                  | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                  | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr,
                (uint32_t)bufBarriers.size(), bufBarriers.data(),
                0, nullptr);
        }
    }

    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBegin.renderPass = s.renderPass;
    rpBegin.framebuffer = renderTarget.framebuffer;
    rpBegin.renderArea = {{0, 0}, {(uint32_t)width, (uint32_t)height}};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;
    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // Y-flip viewport: shaders use OpenGL-style NDC (+Y up); Vulkan framebuffer
    // origin is top-left. Negative height inverts Y so the math stays GL-style.
    VkViewport viewport = {0, (float)height, (float)width, -(float)height, 0.0f, 1.0f};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor = {{0, 0}, {(uint32_t)width, (uint32_t)height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        s.pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    LudusPushConstants pc = {};
    pc.u_width_polyline_regular    = s.widthPolylineRegular > 0 ? s.widthPolylineRegular : 7.0f;
    pc.u_width_polyline_bev        = s.widthPolylineBev     > 0 ? s.widthPolylineBev     : 4.0f;
    pc.u_width_ego_traj_regular    = s.widthEgoTrajRegular  > 0 ? s.widthEgoTrajRegular  : 12.0f;
    pc.u_width_ego_traj_bev        = s.widthEgoTrajBev      > 0 ? s.widthEgoTrajBev      : 5.0f;
    pc.u_width_wireframe           = s.widthWireframe       > 0 ? s.widthWireframe       : 2.0f;
    pc.u_resolution_scale          = s.resolutionScale      > 0 ? s.resolutionScale      : 1.0f;
    pc.u_depth_scaling             = s.depthScaling;
    pc.u_max_extrapolation_us      = s.maxExtrapolationUs;
    pc.u_color_palette_size        = s.colorPaletteSize;
    pc.u_num_queries               = numQueries;
    pc.u_tessellation_threshold    = s.tessellationThreshold;
    pc.u_max_tessellation_polyline = s.maxTessellationLevelPolyline;
    pc.u_max_tessellation_polygon  = s.maxTessellationLevelPolygon;
    pc.u_max_tessellation_cube     = s.maxTessellationLevelCube;
    pc.u_cull_radius_scale         = s.cullRadiusScale;
    pc.u_fog_enabled               = s.depthScaling;
    pc.u_output_width              = (uint32_t)width;
    pc.u_output_height             = (uint32_t)height;

    const char* dbg = getenv("LUDUS_VK_PIPELINES");
    bool draw_polyline = !dbg || strchr(dbg, 'P');
    bool draw_polygon  = !dbg || strchr(dbg, 'G');
    bool draw_obstacle = !dbg || strchr(dbg, 'O');

    auto drawMeshTasks = s.vkctx.pfnCmdDrawMeshTasksEXT;

    if (draw_polyline && s.polylinePoolUsed > 0 && drawMeshTasks && s.pipelinePolyline) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelinePolyline);
        pc.u_num_polyline_pools = std::max(1u, (uint32_t)s.maxPolylinePoolsPerScene);
        pc.u_max_varrays_per_pool = std::max(1u, (uint32_t)s.maxVarraysPerTsPolyline);
        pc.u_cube_pool_index = 0;
        vkCmdPushConstants(cmd, s.pipelineLayout,
            kLudusPushConstantStages, 0, sizeof(pc), &pc);
        uint32_t totalWG = numQueries * pc.u_num_polyline_pools * pc.u_max_varrays_per_pool;
        drawMeshTasks(cmd, totalWG, 1, 1);
    }

    if (draw_polygon && s.polygonPoolUsed > 0 && drawMeshTasks && s.pipelinePolygon) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelinePolygon);
        pc.u_num_polygon_pools = std::max(1u, (uint32_t)s.maxPolygonPoolsPerScene);
        pc.u_max_varrays_per_pool = std::max(1u, (uint32_t)s.maxVarraysPerTsPolygon);
        pc.u_cube_pool_index = 0;
        vkCmdPushConstants(cmd, s.pipelineLayout,
            kLudusPushConstantStages, 0, sizeof(pc), &pc);
        uint32_t totalWG = numQueries * pc.u_num_polygon_pools * pc.u_max_varrays_per_pool;
        drawMeshTasks(cmd, totalWG, 1, 1);
    }

    uint32_t maxObstacles = std::max(1u, (uint32_t)s.maxObstaclesPerPool);
    for (int poolIdx = 0; poolIdx < s.maxCubePoolsPerScene; poolIdx++) {
        if (!(draw_obstacle && drawMeshTasks && s.pipelineObstacle)) break;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.pipelineObstacle);
        pc.u_max_obstacles = maxObstacles;
        pc.u_cube_pool_index = poolIdx;
        // Wireframe rendering is driven by the per-pool ObstaclePool.render_flags
        // SSBO field that the mesh shader reads (CUBE_FLAG_WIREFRAME bit).
        vkCmdPushConstants(cmd, s.pipelineLayout,
            kLudusPushConstantStages, 0, sizeof(pc), &pc);
        uint32_t totalWG = numQueries * maxObstacles;
        drawMeshTasks(cmd, totalWG, 1, 1);
    }

    vkCmdEndRenderPass(cmd);

    // Fixed-function color/depth operations are ordered, unlike fragment SSBO
    // stores. Publish the resolved color attachment to one compute pass that
    // writes the CUDA-visible linear buffer. This is the only steady-state
    // intra-submission synchronization point added by the export path.
    VkImageMemoryBarrier colorToCompute = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    colorToCompute.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    colorToCompute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    colorToCompute.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorToCompute.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorToCompute.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorToCompute.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    colorToCompute.image = renderTarget.colorAttachment.image;
    colorToCompute.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorToCompute.subresourceRange.baseMipLevel = 0;
    colorToCompute.subresourceRange.levelCount = 1;
    colorToCompute.subresourceRange.baseArrayLayer = 0;
    colorToCompute.subresourceRange.layerCount = numQueries;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &colorToCompute);

    vkCmdBindPipeline(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipelineExport);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        s.pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmd, s.pipelineLayout, kLudusPushConstantStages,
        0, sizeof(pc), &pc);
    vkCmdDispatch(cmd,
        ((uint32_t)width + 15u) / 16u,
        ((uint32_t)height + 15u) / 16u,
        (uint32_t)numQueries);

    // Compute has populated the exported SSBO; release it and the input
    // allocations to CUDA in the existing timeline handoff.
    std::vector<VkBufferMemoryBarrier> releaseBarriers;
    auto releaseToCuda = [&](VkExternalBuffer& buffer, VkAccessFlags srcAccess) {
        if (buffer.buffer == VK_NULL_HANDLE || buffer.size == 0 || !buffer.cuExtMem)
            return;
        VkBufferMemoryBarrier barrier = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = 0;
        barrier.srcQueueFamilyIndex = s.vkctx.graphicsQueueFamily;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        barrier.buffer = buffer.buffer;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        releaseBarriers.push_back(barrier);
    };
    releaseToCuda(s.timestampsBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.int32Buffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.vertexBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.triangleBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.poseBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.floatBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.sceneBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.polylinePoolBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.polygonPoolBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.obstaclePoolBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.colorPaletteBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.cameraIntrinsicsBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.cameraPoseBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(s.queryBuffer, VK_ACCESS_SHADER_READ_BIT);
    releaseToCuda(outputBuffer, VK_ACCESS_SHADER_WRITE_BIT);

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT
          | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT
          | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
          | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr,
        (uint32_t)releaseBarriers.size(), releaseBarriers.data(),
        0, nullptr);

    VK_CHECK(vkEndCommandBuffer(cmd));

    const uint64_t vulkanDoneValue = ++s.vkctx.nextTimelineValue;
    VkTimelineSemaphoreSubmitInfo timelineSubmit = {
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timelineSubmit.waitSemaphoreValueCount = 1;
    timelineSubmit.pWaitSemaphoreValues = &cudaReadyValue;
    timelineSubmit.signalSemaphoreValueCount = 1;
    timelineSubmit.pSignalSemaphoreValues = &vulkanDoneValue;
    VkPipelineStageFlags waitStage =
        VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT
        | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT
        | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.pNext = &timelineSubmit;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &s.vkctx.interopTimeline;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &s.vkctx.interopTimeline;
    VK_CHECK(vkQueueSubmit(s.vkctx.graphicsQueue, 1, &submitInfo, frameFence));
    s.lastVulkanDoneValue = vulkanDoneValue;
    exportSlot.releasedToCuda = 1;
}

//=============================================================================
// Copy Results
//=============================================================================

uint8_t* ludusMapBatchResultsVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    int outputSlot, int width, int height, int numQueries)
{
    (void)nvdr_ctx;
    TORCH_CHECK(outputSlot >= 0 && outputSlot < LUDUS_VK_OUTPUT_SLOTS,
        "invalid Vulkan output slot ", outputSlot);
    const size_t bytes = (size_t)width * height * 4 * numQueries;
    LudusVkExportSlot& exportSlot = s.exportSlots[outputSlot];
    TORCH_CHECK(exportSlot.width == width &&
        exportSlot.height == height && exportSlot.layers == numQueries,
        "Vulkan export slot shape changed before mapping");
    VkExternalBuffer& output = exportSlot.linearRgba;
    TORCH_CHECK(output.cuDevPtr != 0 && output.size >= bytes,
        "Vulkan output SSBO is not CUDA-importable or is undersized");
    waitInteropTimelineOnCuda(s.vkctx, stream, s.lastVulkanDoneValue);
    return reinterpret_cast<uint8_t*>(output.cuDevPtr);
}

void ludusCopyBatchResultsVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    int outputSlot, uint8_t* outputPtr,
    int width, int height, int numQueries)
{
    size_t totalSize = (size_t)width * height * 4 * numQueries;
    uint8_t* sharedOutput = ludusMapBatchResultsVk(
        NVDR_CTX_PARAMS, s, stream, outputSlot,
        width, height, numQueries);
    // Legacy staging/JPEG entry points request an owned CUDA copy. The primary
    // torch render binding bypasses this function and wraps sharedOutput.
    cudaError_t copyError = cudaMemcpyAsync(
        outputPtr,
        sharedOutput,
        totalSize,
        cudaMemcpyDeviceToDevice,
        stream);
    TORCH_CHECK(copyError == cudaSuccess,
        "cudaMemcpyAsync from Vulkan shared output failed: ", (int)copyError);
}

int ludusCopyBatchResultsToStagingVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, cudaStream_t stream,
    int outputSlot, int idx, int width, int height, int numQueries)
{
    TORCH_CHECK(idx >= 0 && idx < 2,
        "invalid Vulkan staging slot ", idx);
    TORCH_CHECK(s.stagingValid[idx],
        "Vulkan staging slot ", idx, " was not reserved");

    size_t totalSize = (size_t)width * height * 4 * numQueries;
    if (totalSize > s.stagingBufferSize[idx]) {
        // Only resize the selected slot. The other slot may still be pending
        // retrieval with different dimensions and must remain untouched.
        if (s.stagingValid[idx])
            cudaEventSynchronize(s.stagingReadyEvent[idx]);
        if (s.stagingBuffer[idx]) cudaFree(s.stagingBuffer[idx]);
        cudaMalloc(&s.stagingBuffer[idx], totalSize);
        s.stagingBufferSize[idx] = totalSize;
    }

    ludusCopyBatchResultsVk(
        NVDR_CTX_PARAMS, s, stream, outputSlot, s.stagingBuffer[idx],
        width, height, numQueries);
    cudaEventRecord(s.stagingReadyEvent[idx], stream);

    s.stagingWidth[idx] = width;
    s.stagingHeight[idx] = height;
    s.stagingNumQueries[idx] = numQueries;
    return idx;
}

void ludusCopyStagingToOutputVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx,
    uint8_t* outputPtr)
{
    (void)nvdr_ctx;
    TORCH_CHECK(stagingIdx >= 0 && stagingIdx < 2,
        "invalid Vulkan staging slot ", stagingIdx);
    TORCH_CHECK(s.stagingValid[stagingIdx],
        "Vulkan staging slot ", stagingIdx, " has no pending frame");
    cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]);
    size_t size = (size_t)s.stagingWidth[stagingIdx]
        * s.stagingHeight[stagingIdx] * 4
        * s.stagingNumQueries[stagingIdx];
    cudaMemcpy(outputPtr, s.stagingBuffer[stagingIdx], size, cudaMemcpyDeviceToDevice);
}

int ludusStartAsyncHostTransferVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx)
{
    (void)nvdr_ctx;
    TORCH_CHECK(stagingIdx >= 0 && stagingIdx < 2,
        "invalid Vulkan staging slot ", stagingIdx);
    if (!s.stagingValid[stagingIdx]) return -1;

    int pinnedIdx = s.currentPinnedIdx;
    s.currentPinnedIdx = 1 - pinnedIdx;

    const int width = s.stagingWidth[stagingIdx];
    const int height = s.stagingHeight[stagingIdx];
    const int numQueries = s.stagingNumQueries[stagingIdx];
    size_t size = (size_t)width * height * 4 * numQueries;
    if (size > s.pinnedHostBufferSize[pinnedIdx]) {
        // Preserve the other pinned result while this slot grows.
        if (s.pinnedValid[pinnedIdx])
            cudaEventSynchronize(s.pinnedReadyEvent[pinnedIdx]);
        if (s.pinnedHostBuffer[pinnedIdx])
            cudaFreeHost(s.pinnedHostBuffer[pinnedIdx]);
        cudaMallocHost(&s.pinnedHostBuffer[pinnedIdx], size);
        s.pinnedHostBufferSize[pinnedIdx] = size;
        s.pinnedValid[pinnedIdx] = 0;
    }

    cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]);
    cudaMemcpyAsync(s.pinnedHostBuffer[pinnedIdx], s.stagingBuffer[stagingIdx],
        size, cudaMemcpyDeviceToHost, s.copyStream);
    cudaEventRecord(s.pinnedReadyEvent[pinnedIdx], s.copyStream);

    s.pinnedWidth[pinnedIdx] = width;
    s.pinnedHeight[pinnedIdx] = height;
    s.pinnedNumQueries[pinnedIdx] = numQueries;
    s.pinnedValid[pinnedIdx] = 1;

    return pinnedIdx;
}

int ludusIsPinnedBufferReadyVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int pinnedIdx)
{
    (void)nvdr_ctx;
    if (!s.pinnedValid[pinnedIdx]) return 0;
    return (cudaEventQuery(s.pinnedReadyEvent[pinnedIdx]) == cudaSuccess) ? 1 : 0;
}

int ludusIsHostTransferCompleteVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s)
{
    int prev = 1 - s.currentPinnedIdx;
    return ludusIsPinnedBufferReadyVk(NVDR_CTX_PARAMS, s, prev);
}

int ludusEncodeJpegBatchStagingVk(
    NVDR_CTX_ARGS, LudusTimestampedVkState& s, int stagingIdx, int quality,
    std::vector<std::pair<uint8_t*, size_t>>& outJpegs)
{
    (void)nvdr_ctx;
    TORCH_CHECK(stagingIdx >= 0 && stagingIdx < 2,
        "invalid Vulkan staging slot ", stagingIdx);
    TORCH_CHECK(s.stagingValid[stagingIdx],
        "Vulkan staging slot ", stagingIdx, " has no pending frame");
    if (!s.nvjpegInitialized) {
        nvjpegCreateSimple(&s.nvjpegHandle);
        nvjpegEncoderStateCreate(s.nvjpegHandle, &s.nvjpegEncoderState, 0);
        nvjpegEncoderParamsCreate(s.nvjpegHandle, &s.nvjpegEncoderParams, 0);
        s.nvjpegInitialized = 1;
    }
    nvjpegEncoderParamsSetQuality(s.nvjpegEncoderParams, quality, 0);

    cudaEventSynchronize(s.stagingReadyEvent[stagingIdx]);

    int w = s.stagingWidth[stagingIdx];
    int h = s.stagingHeight[stagingIdx];
    int n = s.stagingNumQueries[stagingIdx];
    size_t layerSize = (size_t)w * h * 4;
    size_t rgbSize = (size_t)w * h * 3;

    if (rgbSize > s.jpegFlipBufferSize) {
        if (s.jpegFlipBuffer) cudaFree(s.jpegFlipBuffer);
        cudaMalloc(&s.jpegFlipBuffer, rgbSize);
        s.jpegFlipBufferSize = rgbSize;
    }

    outJpegs.resize(n);
    for (int i = 0; i < n; i++) {
        uint8_t* srcRgba = s.stagingBuffer[stagingIdx] + i * layerSize;
        launchRgbaToRgbFlip(srcRgba, s.jpegFlipBuffer, w, h, 0);
        cudaDeviceSynchronize();

        nvjpegImage_t img;
        memset(&img, 0, sizeof(img));
        img.channel[0] = s.jpegFlipBuffer;
        img.pitch[0] = w * 3;

        nvjpegEncodeImage(s.nvjpegHandle, s.nvjpegEncoderState, s.nvjpegEncoderParams,
            &img, NVJPEG_INPUT_RGBI, w, h, 0);

        size_t jpegSize = 0;
        nvjpegEncodeRetrieveBitstream(s.nvjpegHandle, s.nvjpegEncoderState, nullptr, &jpegSize, 0);

        if (jpegSize > s.jpegOutputBufferSize) {
            if (s.jpegOutputBuffer) cudaFreeHost(s.jpegOutputBuffer);
            cudaMallocHost(&s.jpegOutputBuffer, jpegSize);
            s.jpegOutputBufferSize = jpegSize;
        }

        nvjpegEncodeRetrieveBitstream(s.nvjpegHandle, s.nvjpegEncoderState,
            s.jpegOutputBuffer, &jpegSize, 0);

        uint8_t* jpegCopy = (uint8_t*)malloc(jpegSize);
        memcpy(jpegCopy, s.jpegOutputBuffer, jpegSize);
        outJpegs[i] = {jpegCopy, jpegSize};
    }

    return n;
}

//=============================================================================
// Cleanup
//=============================================================================

void ludusClearScenesVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s)
{
    (void)nvdr_ctx;
    s.numScenes = 0;
    s.timestampsUsed = s.int32Used = s.vertexUsed = s.triangleUsed = 0;
    s.poseUsed = s.floatUsed = 0;
    s.polylinePoolUsed = s.polygonPoolUsed = s.obstaclePoolUsed = 0;
    s.maxObstaclesPerPool = s.maxCubePoolsPerScene = 0;
    s.maxPolylinePoolsPerScene = s.maxPolygonPoolsPerScene = 0;
    s.maxVarraysPerTsPolyline = s.maxVarraysPerTsPolygon = 0;
    s.sceneBuffersDirty = 1;
}

void ludusTimestampedReleaseVk(NVDR_CTX_ARGS, LudusTimestampedVkState& s)
{
    (void)nvdr_ctx;
    if (s.vkctx.device) vkDeviceWaitIdle(s.vkctx.device);

    for (int i = 0; i < 2; i++) {
        if (s.stagingBuffer[i]) cudaFree(s.stagingBuffer[i]);
        if (s.pinnedHostBuffer[i]) cudaFreeHost(s.pinnedHostBuffer[i]);
        if (s.stagingReadyEvent[i]) cudaEventDestroy(s.stagingReadyEvent[i]);
        if (s.pinnedReadyEvent[i]) cudaEventDestroy(s.pinnedReadyEvent[i]);
    }
    if (s.copyStream) cudaStreamDestroy(s.copyStream);
    if (s.jpegOutputBuffer) cudaFreeHost(s.jpegOutputBuffer);
    if (s.jpegFlipBuffer) cudaFree(s.jpegFlipBuffer);
    if (s.nvjpegInitialized) {
        nvjpegEncoderParamsDestroy(s.nvjpegEncoderParams);
        nvjpegEncoderStateDestroy(s.nvjpegEncoderState);
        nvjpegDestroy(s.nvjpegHandle);
    }

    for (int i = 0; i < 10; i++)
        if (s.shaderModules[i]) vkDestroyShaderModule(s.vkctx.device, s.shaderModules[i], nullptr);

    if (s.pipelinePolyline) vkDestroyPipeline(s.vkctx.device, s.pipelinePolyline, nullptr);
    if (s.pipelinePolygon)  vkDestroyPipeline(s.vkctx.device, s.pipelinePolygon, nullptr);
    if (s.pipelineObstacle) vkDestroyPipeline(s.vkctx.device, s.pipelineObstacle, nullptr);
    if (s.pipelineExport)   vkDestroyPipeline(s.vkctx.device, s.pipelineExport, nullptr);
    if (s.pipelineLayout)   vkDestroyPipelineLayout(s.vkctx.device, s.pipelineLayout, nullptr);
    if (s.descriptorPool)   vkDestroyDescriptorPool(s.vkctx.device, s.descriptorPool, nullptr);
    if (s.descriptorSetLayout) vkDestroyDescriptorSetLayout(s.vkctx.device, s.descriptorSetLayout, nullptr);

    destroyRenderTargets(s);
    if (s.renderPass)  vkDestroyRenderPass(s.vkctx.device, s.renderPass, nullptr);

    destroyExternalBuffer(s.vkctx, s.timestampsBuffer);
    destroyExternalBuffer(s.vkctx, s.int32Buffer);
    destroyExternalBuffer(s.vkctx, s.vertexBuffer);
    destroyExternalBuffer(s.vkctx, s.triangleBuffer);
    destroyExternalBuffer(s.vkctx, s.poseBuffer);
    destroyExternalBuffer(s.vkctx, s.floatBuffer);
    destroyExternalBuffer(s.vkctx, s.sceneBuffer);
    destroyExternalBuffer(s.vkctx, s.polylinePoolBuffer);
    destroyExternalBuffer(s.vkctx, s.polygonPoolBuffer);
    destroyExternalBuffer(s.vkctx, s.obstaclePoolBuffer);
    destroyExternalBuffer(s.vkctx, s.colorPaletteBuffer);
    destroyExternalBuffer(s.vkctx, s.cameraIntrinsicsBuffer);
    destroyExternalBuffer(s.vkctx, s.cameraPoseBuffer);
    destroyExternalBuffer(s.vkctx, s.queryBuffer);
    for (int i = 0; i < LUDUS_VK_OUTPUT_SLOTS; ++i)
        destroyExternalBuffer(s.vkctx, s.exportSlots[i].linearRgba);

    destroyVkContext(s.vkctx);
    memset(&s, 0, sizeof(s));
}
