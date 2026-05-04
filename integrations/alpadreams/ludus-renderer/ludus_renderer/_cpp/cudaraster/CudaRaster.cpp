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

#include "Defs.hpp"
#include "CudaRaster.hpp"
#include "RasterImpl.hpp"

using namespace CR;

//------------------------------------------------------------------------
// Stub interface implementation.
//------------------------------------------------------------------------

CudaRaster::CudaRaster()
{
    m_impl = new RasterImpl();
}

CudaRaster::~CudaRaster()
{
    delete m_impl;
}

void CudaRaster::setBufferSize(int width, int height, int numImages)
{
    m_impl->setBufferSize(Vec3i(width, height, numImages));
}

void CudaRaster::setViewport(int width, int height, int offsetX, int offsetY)
{
    m_impl->setViewport(Vec2i(width, height), Vec2i(offsetX, offsetY));
}

void CudaRaster::setRenderModeFlags(U32 flags)
{
    m_impl->setRenderModeFlags(flags);
}

void CudaRaster::deferredClear(U32 clearColor)
{
    m_impl->deferredClear(clearColor);
}

void CudaRaster::setVertexBuffer(void* vertices, int numVertices)
{
    m_impl->setVertexBuffer(vertices, numVertices);
}

void CudaRaster::setIndexBuffer(void* indices, int numTriangles)
{
    m_impl->setIndexBuffer(indices, numTriangles);
}

void CudaRaster::setTiebreakerColorBuffer(void* colors)
{
    m_impl->setTiebreakerColorBuffer(colors);
}

void CudaRaster::setDeterministicTiebreaker(bool enable)
{
    m_impl->setDeterministicTiebreaker(enable);
}

bool CudaRaster::drawTriangles(const int* ranges, bool peel, cudaStream_t stream)
{
    return m_impl->drawTriangles((const Vec2i*)ranges, peel, stream);
}

void* CudaRaster::getColorBuffer(void)
{
    return m_impl->getColorBuffer();
}

void* CudaRaster::getDepthBuffer(void)
{
    return m_impl->getDepthBuffer();
}

void CudaRaster::swapDepthAndPeel(void)
{
    m_impl->swapDepthAndPeel();
}

int CudaRaster::getBufferWidth(void) const
{
    return m_impl->getBufferWidth();
}

int CudaRaster::getBufferHeight(void) const
{
    return m_impl->getBufferHeight();
}

int CudaRaster::getNumImages(void) const
{
    return m_impl->getNumImages();
}

//------------------------------------------------------------------------
