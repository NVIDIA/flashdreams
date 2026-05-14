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

#include <cuda_runtime.h>
#include <stdint.h>

// Kernel to convert RGBA to RGB with vertical flip
// Input: RGBA image (4 bytes/pixel), bottom-to-top (OpenGL)
// Output: RGB image (3 bytes/pixel), top-to-bottom (JPEG)
__global__ void rgbaToRgbFlipKernel(
    const uint8_t* __restrict__ srcRgba,
    uint8_t* __restrict__ dstRgb,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    // Flip: dst row y comes from src row (height - 1 - y)
    int srcRow = height - 1 - y;
    
    // Read RGBA from source (4 bytes per pixel)
    int srcIdx = (srcRow * width + x) * 4;
    uint8_t r = srcRgba[srcIdx + 0];
    uint8_t g = srcRgba[srcIdx + 1];
    uint8_t b = srcRgba[srcIdx + 2];
    // Alpha ignored
    
    // Write RGB to destination (3 bytes per pixel)
    int dstIdx = (y * width + x) * 3;
    dstRgb[dstIdx + 0] = r;
    dstRgb[dstIdx + 1] = g;
    dstRgb[dstIdx + 2] = b;
}

// Host function to launch the kernel
extern "C" void launchRgbaToRgbFlip(
    const uint8_t* srcRgba,
    uint8_t* dstRgb,
    int width,
    int height,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    rgbaToRgbFlipKernel<<<grid, block, 0, stream>>>(srcRgba, dstRgb, width, height);
}
