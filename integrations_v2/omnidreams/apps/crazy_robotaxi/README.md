<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OmniDreams Crazy Robotaxi

Install the OmniDreams integration and launch its Crazy Robotaxi application:

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 crazy-robotaxi-omnidreams \
  --mode native-window
```

Model assets download from Hugging Face on first use. Export `HF_TOKEN` when
the selected repository requires authentication.

Available application slugs:

| Application slug | Pipeline config |
| --- | --- |
| `crazy-robotaxi-omnidreams` | `OMNIDREAMS_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-optimized-gb300` | `OMNIDREAMS_OPTIMIZED_GB300_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000` | `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-perf` | `OMNIDREAMS_PERF_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-fast-perf` | `OMNIDREAMS_FAST_PERF_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-responsive` | `OMNIDREAMS_RESPONSIVE_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-optimized-gb300-responsive` | `OMNIDREAMS_OPTIMIZED_GB300_RESPONSIVE_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000-responsive` | `OMNIDREAMS_OPTIMIZED_RTX_PRO_6000_RESPONSIVE_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-perf-responsive` | `OMNIDREAMS_PERF_RESPONSIVE_PIPELINE_CONFIG` |
| `crazy-robotaxi-omnidreams-fast-perf-responsive` | `OMNIDREAMS_FAST_PERF_RESPONSIVE_PIPELINE_CONFIG` |

The non-responsive `perf` variants require a one-time preparation step:

```bash
uv run --no-sync omnidreams-prepare --perf
```

See the shared [Crazy Robotaxi README](../../../../apps/crazy_robotaxi/README.md)
for controls, application arguments, game modes, and tests. See the
[OmniDreams integration README](../../README.md) for model details.
