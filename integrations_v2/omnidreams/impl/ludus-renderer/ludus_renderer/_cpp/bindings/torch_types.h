// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "torch_common.inl"
#include <mutex>
#include <optional>

//------------------------------------------------------------------------
// Forward declarations.

struct LudusCudaState;

//------------------------------------------------------------------------
// Python Ludus CUDA state wrapper.

class LudusCudaStateWrapper
{
public:
    LudusCudaStateWrapper       (int cudaDeviceIdx);
    ~LudusCudaStateWrapper      (void);

    void setLineWidths          (float polyline_regular, float polyline_bev,
                                 float ego_traj_regular, float ego_traj_bev,
                                 float wireframe);
    void setResolutionScale     (float scale);
    void setWidthInNdc          (bool enabled);
    void setDepthFade           (bool enabled);
    void setCullBehindCamera    (bool enabled);
    void setDepthScaling        (float enabled);
    void setCullRadius          (float radius);
    void setMaxTessellationLevels(int polyline, int polygon, int cube);
    void uploadColorPalette     (torch::Tensor colors);
    void uploadWidthTable       (torch::Tensor widths);
    void setMsaaSamples         (int samples);

    LudusCudaState*             pState;
    int                         cudaDeviceIdx;
    cudaEvent_t                 lastUseEvent;
    bool                        hasLastUseEvent;
    std::mutex                  stateMutex;
};

//------------------------------------------------------------------------
// Python CudaRaster API test wrapper.

class CudaRasterTestWrapper
{
public:
    CudaRasterTestWrapper       (int cudaDeviceIdx);
    ~CudaRasterTestWrapper      (void);

    void setBufferSize          (int width, int height, int numImages);
    void setViewport            (int width, int height, int offsetX, int offsetY);
    void setRenderModeFlags     (unsigned int flags);
    void deferredClear          (unsigned int clearColor);
    void setVertexBuffer        (torch::Tensor vertices);
    void setIndexBuffer         (torch::Tensor indices);
    void setTiebreakerColorBuffer(torch::Tensor colors);
    void setDeterministicTiebreaker(bool enable);
    bool drawTriangles          (std::optional<torch::Tensor> ranges, bool peel);
    void swapDepthAndPeel       (void);
    torch::Tensor getColorBuffer(void);
    torch::Tensor getDepthBuffer(void);
    int getBufferWidth          (void) const;
    int getBufferHeight         (void) const;
    int getNumImages            (void) const;

private:
    class Impl;
    Impl*                       m_impl;
};

//------------------------------------------------------------------------
