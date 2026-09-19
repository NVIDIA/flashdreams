<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams

OmniDreams is an HD-map-conditioned streaming driving world-model integration.
Its public model variants are `OmnidreamsPipelineConfig` literals in
`omnidreams.config`; reusable applications own interactive I/O.

## Pipeline configurations

- `OMNIDREAMS_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG`
- `OMNIDREAMS_PERF_PIPELINE_CONFIG`
- `OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG`
- `OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG`
- `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG`

## Install

```bash
uv sync --package flashdreams-omnidreams --inexact
```

Checkpoints and example scenes download from Hugging Face on first use. Export
`HF_TOKEN` when the selected repository requires authentication.

Native third-party sources download on first use into
`artifacts/omnidreams/thirdparty`. Set
`FLASHDREAMS_OMNIDREAMS_TRY_THIRDPARTY_RESYNC=1` to attempt a clean redownload
on each load. Do not use resync while another OmniDreams process is using the
sources. The default value is `0` (only downloads missing sources).

## Applications

- [Interactive Drive](apps/interactive_drive/README.md)
- [Crazy Robotaxi](apps/crazy_robotaxi/README.md)

Launch Interactive Drive in a browser with:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089
```

## Programmatic pipeline access

```python
from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG

pipeline = OMNIDREAMS_PIPELINE_CONFIG.setup().to("cuda").eval()
```

## Layer-wise DiT offload

Layer-wise offload is an opt-in, GPU-memory-saving inference mode. It keeps the
OmniDreams DiT block parameters in pinned CPU memory and streams the current and
next blocks through two reusable CUDA staging slots. The
`diffusion_model.transformer.enable_layerwise_offload` flag defaults to `False`.
Enable it by deriving from the regular pipeline configuration before
constructing the pipeline:

```python
from flashdreams.infra.config import derive_config
from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG

offload_config = derive_config(
    OMNIDREAMS_PIPELINE_CONFIG,
    name="omnidreams-layerwise-offload",
    diffusion_model=dict(
        transformer=dict(enable_layerwise_offload=True),
    ),
)
pipeline = offload_config.setup().to("cuda").eval()
```

The regular configuration already uses the required OmniDreams self- and
cross-attention backends with native DiT acceleration disabled. Do not enable
the flag directly on the native `perf` or optimized-attention configurations.
Layer-wise offload automatically bypasses whole-DiT `torch.compile` and CUDA
Graph execution, and it does not support runtime text-edit LoRA, live-edit drift
correction, training, post-construction dtype conversion, or state-dict
operations. Construct the pipeline before moving it to CUDA, as shown above.
Expect additional pinned host memory and host-to-device traffic in exchange for
lower GPU memory use.

For Interactive Drive, select the regular slug to leave the flag disabled or
the layer-wise-offload slug to enable it when the pipeline is constructed:

```bash
# Off (the default)
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089

# On
uv run --no-sync flashdreams-run-v2 \
  interactive-drive-omnidreams-layerwise-offload \
  --mode webrtc --host 0.0.0.0 --port 8089
```

This is a startup-time switch rather than a live UI toggle: changing it requires
reconstructing the pipeline because setup replaces inactive block storage with
empty placeholders.

## Tests

```bash
uv sync --package flashdreams-omnidreams --extra dev --group test --inexact
uv run --no-sync pytest integrations_v2/omnidreams/tests -m ci_cpu
```
