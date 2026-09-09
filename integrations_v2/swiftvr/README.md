<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SwiftVR

[SwiftVR](https://github.com/H-oliday/SwiftVR) real-time, one-step streaming
video restoration packaged as a FlashDreams postprocessor. The integration
uses the upstream ReAE streaming protocol and mask-free shifted-window
attention on FlashDreams' native WAN transformer components. Upstream
Diffusers-format weights are remapped at load time; Diffusers is not a runtime
dependency.

The heavyweight checkpoint is loaded and prewarmed by the first session, then
stays resident across replacement sessions. Each session owns only its causal
temporal state.

## Use as a postprocessor

Install the workspace package and select one preset:

```bash
uv sync --package flashdreams-swiftvr --inexact
uv run --no-sync flashdreams-run-v2 <application> \
  --postprocess-preset swiftvr-4x
```

The `swiftvr-4x` preset accepts RGB video in any FlashDreams-supported tensor
layout and returns the same number of frames at four times the input width and
height. The checkpoint downloads from `H-oliday/SwiftVR` on first use.

Programmatic configuration stays small:

```python
from swiftvr.impl.postprocess import SwiftVRPostProcessorConfig

postprocessor = SwiftVRPostProcessorConfig(
    scale=4,
    chunk_size=24,
    prewarm=True,
)
```

Set `checkpoint` to a local directory for offline use. `chunk_size` must be a
multiple of four. `dit_overlap=0` is the upstream throughput path;
`dit_overlap=1` trades speed for latent overlap blending. `compile_blocks` is
off by default because it adds a long one-time compilation phase. Loading the
upstream FP32 transformer remaps roughly 19 GiB of weights in host memory before
moving them to the selected CUDA dtype; both single-file and standard sharded
safetensors checkpoints are accepted.

## End-to-end V2V demo

The integration binds the reusable V2V application:

```bash
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --output-path artifacts/swiftvr.mp4 -- \
  --video-path input.mp4
```

For a reproducible QHD throughput run, use a 640x360, 30 FPS input and collect
runtime stats after prewarming:

```bash
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --output-path artifacts/swiftvr-qhd.mp4 \
  --stats-path artifacts/swiftvr-qhd.json -- \
  --video-path input-640x360-30fps.mp4
```

The output is 2560x1440 at the source frame rate. Compare it visually against
the source or a same-resolution reference, and report speed only from complete
steady-state 24-frame chunks; checkpoint loading and prewarm are intentionally
outside those samples.

## Validation

```bash
uv run --package flashdreams-swiftvr --extra dev \
  pytest integrations_v2/swiftvr/tests -m ci_cpu -v
```

The adapted code is pinned in attribution to upstream SwiftVR commit
`5ca168cef6ca7200f135fdfea85e5e13d12c5b53` (Apache-2.0).
