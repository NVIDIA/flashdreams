<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SwiftVR validation report

Measured 2026-09-09 from FlashDreams base `d9b25167` with SwiftVR source
reference `5ca168ce` and Hugging Face snapshot `743ed253`. The host used one
NVIDIA GB300 (256 GB), driver 595.71.05, CUDA 13.2, Torch 2.12.1+cu130,
Diffusers 0.38.0, BF16, PyTorch SDPA, 24-frame chunks, no DiT overlap, and no
`torch.compile`.

## Correctness

- The adapted ReAE has the same 128 state-dict keys and tensor shapes as
  upstream.
- A weighted nine-frame streaming run at 64x36 input / 128x72 output matched
  the upstream SwiftVR output exactly: max absolute error, mean absolute error,
  and RMSE were all `0.0`.
- The V2V MP4 runs retained all 120 input frames, including SwiftVR's delayed
  three-frame causal head and padded tail.

## Throughput

The model was prewarmed before recording five 24-frame chunks. Steady metrics
exclude the first chunk and final flush.

| Output | Steady median | Slowest steady chunk | Full model pass | Peak CUDA allocated |
| --- | ---: | ---: | ---: | ---: |
| 1920x1080 | 103.27 FPS | 99.16 FPS | 98.43 FPS | not recorded |
| 2560x1440 | 60.91 FPS | 59.74 FPS | 58.54 FPS | 35.69 GiB |

The cold QHD V2V command took 31.50 s wall time for 120 frames. That number
includes checkpoint load, two-chunk/tail prewarm, source decode, and MP4 encode;
it is not streaming throughput. First-session prepare took 15.56 s. A second
session in the same process reused the same processor and resident weights:
prepare fell to 0.088 ms and its 24-frame process-plus-flush took 0.448 s.

## Quality demo

A five-second Big Buck Bunny excerpt at 1920x1080 was Lanczos-downsampled to
480x270 and restored at 4x. The output was compared with both the source and a
plain bilinear 4x baseline.

| Method | PSNR | SSIM |
| --- | ---: | ---: |
| Bilinear 4x | 32.10 dB | 0.9144 |
| SwiftVR 4x | 30.71 dB | 0.9040 |

These distortion metrics favor the smoother bilinear result; they do not
measure whether generated detail looks more realistic. In the saved visual
crop, SwiftVR reconstructs visibly sharper grass, leaves, and subject edges.
Treat the image/video comparison as the perceptual demonstration and the table
as an honest fidelity check, not as evidence that SwiftVR improves PSNR.

Local artifacts from this run are under `artifacts/benchmark/swiftvr/`:

- `quality-crop.jpg`: bilinear, SwiftVR, and reference crop.
- `quality-frame.jpg`: full-frame comparison.
- `bilinear-vs-swiftvr.mp4`: five-second side-by-side playback.

The source clip and generated full-resolution MP4s are intentionally not added
to git.
