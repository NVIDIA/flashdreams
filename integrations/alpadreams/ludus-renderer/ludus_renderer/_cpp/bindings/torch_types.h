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
    void setDepthScaling        (float enabled);
    void setCullRadius          (float radius);
    void setMaxTessellationLevels(int polyline, int polygon, int cube);
    void uploadColorPalette     (torch::Tensor colors);
    void setMsaaSamples         (int samples);

    LudusCudaState*             pState;
    int                         cudaDeviceIdx;
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
