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

These throughput numbers predate the native FlashDreams WAN migration described
below; they remain the Diffusers-backed baseline rather than a remeasurement of
the current runtime.

## Correctness

- The shared TAEHV-backed ReAE has the same 128 state-dict keys and tensor
  shapes as upstream; only SwiftVR's Conv3d temporal-growth layer remains
  model-specific.
- A weighted nine-frame streaming run at 64x36 input / 128x72 output matched
  the upstream SwiftVR output exactly: max absolute error, mean absolute error,
  and RMSE were all `0.0`.
- The V2V MP4 runs retained all 120 input frames, including SwiftVR's delayed
  three-frame causal head and padded tail.

### Native FlashDreams WAN parity

Measured 2026-09-09 while migrating the integration from Diffusers to native
FlashDreams WAN components. The legacy reference was captured immediately
before replacing Diffusers. Both paths used snapshot `743ed253`, its prompt
embedding, eager BF16 SDPA, and ran sequentially on one NVIDIA RTX PRO 6000
Blackwell Server Edition (driver 595.80, CUDA 13.0, Torch 2.12.1+cu130).

Each fixture was constructed as
`torch.linspace(-1, 1, 48*T*H*W).reshape(1, 48, T, H, W).bfloat16()` and passed
through the prompt-cache and transformer flow-prediction path at timestep 1000.
The checkpoint remap covered all 825 tensors with no missing, unexpected, or
shape-mismatched keys.

| Latent fixture | Temporal offset | Max absolute error | Mean absolute error | RMSE |
| --- | ---: | ---: | ---: | ---: |
| `1x48x1x8x8` | 7 | 0.01367 | 0.003047 | 0.003916 |
| `1x48x2x34x66` | 3 | 0.01929 | 0.003011 | 0.003885 |

The second fixture patchifies to `2x17x33`, exercising distinct unshifted
(`[0, 16, 17]`) and shifted (`[0, 8, 17]`) window starts, overlap ownership,
and nonzero temporal RoPE. Peak CUDA allocation on that comparison fell from
12.16 GiB for the Diffusers path to 9.69 GiB for the native path. A nine-frame
streaming smoke test also retained all frames at the requested output shape,
and decoding the shifted fixture through the shared ReAE yielded 0.00076 mean
absolute RGB error (0.01172 max, 0.999995 cosine similarity). A one-block
compiled/eager smoke test passed within 0.00782 max absolute error.

The regression limits for reruns are 0.05 max and 0.01 mean absolute latent
error. The captured legacy tensors were saved as
`/tmp/swiftvr-diffusers-transformer-reference.pt` and
`/tmp/swiftvr-diffusers-shifted-reference.pt`; they are run artifacts and are
not tracked. The dependency-free CPU contracts, including the complete
825-key synthetic remap bijection, run with:

```bash
uv run --package flashdreams-swiftvr --extra dev \
  pytest integrations_v2/swiftvr/tests/test_model.py -m ci_cpu -q
```

### Pipeline component parity

Measured 2026-09-10 while splitting the resident runtime into typed streaming
encoder, transformer, and decoder components. The candidate was compared with
a frozen run from pre-refactor commit `7c136801` using snapshot `743ed253`, BF16,
128x128 output, and both zero- and one-latent DiT overlap. Preprocessing,
encoder latents, restored latents, cold/steady/flush decoder chunks, and the
concatenated 17-frame output were bit-for-bit identical. A one-frame input tail
remained buffered until flush in both implementations.

The later shared-TAEHV refactor was checked against a frozen baseline from
commit `330d7f0d`. Standalone encoder cold/steady/tail/flush results, standalone
decoder cold/steady/tail results, and every full-pipeline stage for both overlap
modes remained bit-for-bit identical through the final 17 RGB frames.

The full 30-block `torch.compile` path was rechecked after that refactor. The
concatenated result stayed within the same regression limits (`0.02344` max and
`0.00133` mean absolute error), with all 30 blocks compiled and no graph breaks.
A fresh compiler cache took 22.02 seconds for the two-latent-frame shape and
another 5.47 seconds for the one-latent-frame flush shape; subsequent
small-fixture chunks took 22.13-26.52 ms. These timings validate compilation
behavior and are not a 704p throughput measurement.

## Throughput

### Matched 704p x2 comparison

Both implementations were measured on the RTX PRO 6000 environment above with
the same checkpoint, BF16 dtype, eager execution, and direct in-memory
eight-frame chunks at 1280x704 input and 2560x1408 output.

| Implementation | Median chunk | Median throughput | Slowest chunk | Peak CUDA allocated |
| --- | ---: | ---: | ---: | ---: |
| Diffusers | 354.12 ms | 22.59 FPS | 22.55 FPS | 20.92 GiB |
| FlashDreams native | 300.67 ms | 26.61 FPS | 26.59 FPS | 18.45 GiB |

The native path delivered 1.178x throughput (+17.8%) with 15.1% lower latency
and 2.46 GiB less peak allocation. Two warmup calls and the first chunk of a
fresh stream were discarded before five measured chunks. This isolates SwiftVR
processing and excludes source decode and output encode.

Timing the matching `1x48x2x88x160` latent through the DiT alone attributed the
gain to the WAN migration: Diffusers took 252.20 ms versus 199.26 ms native,
for 1.266x throughput and 21.0% lower latency. DiT peak allocation fell from
12.58 GiB to 10.11 GiB. Each DiT measurement used two warmups and seven samples.

### Diffusers-backed historical baseline

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
