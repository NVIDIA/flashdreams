#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example: GPU-accelerated JPEG encoding with nvjpeg.

This example demonstrates how to use the nvjpeg module to encode
GPU tensors directly to JPEG bytes without CPU copies.
"""

import torch
from pathlib import Path

from ludus_renderer import nvjpeg


def main():
    # Check if nvjpeg is available
    if not nvjpeg.is_available():
        print("nvjpeg is not available on this system")
        return
    
    print("nvjpeg is available!")
    
    # Create output directory
    output_dir = Path("_output/nvjpeg_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Example 1: Encode a batch of random images
    # -------------------------------------------------------------------------
    print("\n--- Example 1: Batch encoding ---")
    
    # Create a batch of 4 random RGB images on GPU
    # Shape: [B, 3, H, W] = [4, 3, 480, 640]
    batch_size = 4
    height, width = 480, 640
    images = torch.randint(0, 256, (batch_size, 3, height, width), 
                           dtype=torch.uint8, device='cuda')
    
    print(f"Input tensor shape: {images.shape}")
    print(f"Input tensor device: {images.device}")
    print(f"Input tensor dtype: {images.dtype}")
    
    # Encode to JPEG with quality 85
    jpeg_list = nvjpeg.encode(images, quality=85)
    
    print(f"Encoded {len(jpeg_list)} images")
    for i, jpeg_bytes in enumerate(jpeg_list):
        output_path = output_dir / f"batch_image_{i}.jpg"
        with open(output_path, 'wb') as f:
            f.write(jpeg_bytes)
        print(f"  Saved {output_path} ({len(jpeg_bytes):,} bytes)")
    
    # -------------------------------------------------------------------------
    # Example 2: Encode a single image
    # -------------------------------------------------------------------------
    print("\n--- Example 2: Single image encoding ---")
    
    # Create a gradient image for visual verification
    # Red increases left to right, Green increases top to bottom
    single_image = torch.zeros((3, height, width), dtype=torch.uint8, device='cuda')
    
    # Create gradients on CPU then copy (for simplicity)
    r_gradient = torch.linspace(0, 255, width).unsqueeze(0).expand(height, width)
    g_gradient = torch.linspace(0, 255, height).unsqueeze(1).expand(height, width)
    b_value = 128  # Constant blue
    
    single_image[0] = r_gradient.to(torch.uint8).cuda()  # Red channel
    single_image[1] = g_gradient.to(torch.uint8).cuda()  # Green channel
    single_image[2] = b_value  # Blue channel
    
    print(f"Input tensor shape: {single_image.shape}")
    
    # Encode single image
    jpeg_bytes = nvjpeg.encode_single(single_image, quality=90)
    
    output_path = output_dir / "gradient_image.jpg"
    with open(output_path, 'wb') as f:
        f.write(jpeg_bytes)
    print(f"Saved {output_path} ({len(jpeg_bytes):,} bytes)")
    
    # -------------------------------------------------------------------------
    # Example 3: Different quality levels
    # -------------------------------------------------------------------------
    print("\n--- Example 3: Quality comparison ---")
    
    for quality in [10, 50, 85, 95]:
        jpeg_bytes = nvjpeg.encode_single(single_image, quality=quality)
        output_path = output_dir / f"quality_{quality}.jpg"
        with open(output_path, 'wb') as f:
            f.write(jpeg_bytes)
        print(f"  Quality {quality:2d}: {len(jpeg_bytes):,} bytes -> {output_path}")
    
    # -------------------------------------------------------------------------
    # Example 4: Benchmark encoding speed
    # -------------------------------------------------------------------------
    print("\n--- Example 4: Benchmark ---")
    
    # Warmup
    for _ in range(5):
        nvjpeg.encode(images, quality=85)
    
    torch.cuda.synchronize()
    
    # Benchmark
    import time
    n_iterations = 100
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        nvjpeg.encode(images, quality=85)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    total_images = n_iterations * batch_size
    images_per_sec = total_images / elapsed
    ms_per_image = (elapsed / total_images) * 1000
    
    print(f"Encoded {total_images} images in {elapsed:.3f}s")
    print(f"Throughput: {images_per_sec:.1f} images/sec")
    print(f"Latency: {ms_per_image:.2f} ms/image")
    
    print(f"\nAll outputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
