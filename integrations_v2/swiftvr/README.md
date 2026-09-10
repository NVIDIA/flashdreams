<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SwiftVR

[SwiftVR](https://github.com/H-oliday/SwiftVR) real-time, one-step streaming
video restoration packaged as a FlashDreams postprocessor. The integration
uses FlashDreams' shared TAEHV encoder/decoder building blocks and decoder cache
with the upstream ReAE temporal-growth layer, plus mask-free shifted-window
attention on FlashDreams' native WAN transformer components. Upstream
Diffusers-format weights are remapped at load time; Diffusers is not a runtime
dependency.

Like FlashVSR, the runtime is a typed `StreamInferencePipeline` composed from
separate streaming encoder, WAN transformer, and streaming decoder components.
Their mutable temporal state lives in per-stream caches while the heavyweight
weights remain resident on the pipeline.

The heavyweight checkpoint is loaded and prewarmed by the first session, then
stays resident across replacement sessions. Each session owns only its causal
temporal state.

## Use as a postprocessor

Install the workspace package and select either the 2x or 4x preset:

```bash
uv sync --package flashdreams-swiftvr --inexact
uv run --no-sync flashdreams-run-v2 <application> \
  --postprocess-preset swiftvr-2x
```

The `swiftvr-2x` and `swiftvr-4x` presets accept RGB video in any
FlashDreams-supported tensor layout and return the same number of frames at two
or four times the input dimensions. The checkpoint downloads from
`H-oliday/SwiftVR` on first use.

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

The V2V binding uses the 2x preset. For a reproducible 704p run, use a
1280x704, 30 FPS input and collect runtime stats after prewarming:

```bash
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --output-path artifacts/swiftvr-1408p.mp4 \
  --stats-path artifacts/swiftvr-1408p.json -- \
  --video-path input-1280x704-30fps.mp4
```

The output is 2560x1408 at the source frame rate. Compare it visually against
the source or a same-resolution reference, and report speed only from complete
steady-state 8-frame chunks; checkpoint loading and prewarm are intentionally
outside those samples.

## Validation

```bash
uv run --package flashdreams-swiftvr --extra dev \
  pytest integrations_v2/swiftvr/tests -m ci_cpu -v
```

The adapted code is pinned in attribution to upstream SwiftVR commit
`5ca168cef6ca7200f135fdfea85e5e13d12c5b53` (Apache-2.0).
