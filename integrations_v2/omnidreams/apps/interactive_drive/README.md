<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams Interactive Drive

Install the OmniDreams integration and launch its Interactive Drive application:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089
```

The default scene and model assets download from Hugging Face on first use.
Export `HF_TOKEN` when the selected repository requires authentication.

Available application slugs:

| Application slug | Pipeline config |
| --- | --- |
| `interactive-drive-omnidreams` | `OMNIDREAMS_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-optimized-gb300` | `OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-optimized-rtx-pro-6000` | `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-perf` | `OMNIDREAMS_PERF_PIPELINE_CONFIG` |
| `interactive-drive-omnidreams-fast-perf` | `OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG` |

The `perf` variants require a one-time preparation step before launch:

```bash
uv run --no-sync omnidreams-prepare --perf
```

See the shared [Interactive Drive README](../../../../apps/interactive_drive/README.md)
for controls, application arguments, output modes, and tests. See the
[OmniDreams integration README](../../README.md) for model details.
