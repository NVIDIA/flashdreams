<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lingbot Cam2V

Install the Lingbot integration and launch its `cam2v-lingbot` application:

```bash
uv sync --package flashdreams-lingbot --inexact
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

Available application slugs:

| Application slug | Pipeline config |
| --- | --- |
| `cam2v-lingbot` | `PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3` |
| `cam2v-lingbot-world-fast` | `PIPELINE_LINGBOT_WORLD_FAST` |
| `cam2v-lingbot-world-fast-taehv-window15-sink3` | `PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3` |
| `cam2v-lingbot-world-v2-14b-causal-fast` | `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST` |
| `cam2v-lingbot-world-v2-14b-causal-fast-taehv-window15-sink3` | `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3` |

`cam2v-lingbot` remains the short compatibility alias for the bounded-window
TAEHV default. All variants use the same Cam2V application defaults.

To load FlashVSR once and expose the **Post-processing** checkbox, start with
`--postprocess-preset flashvsr-v1.1-sparse-1.5 --postprocess-chunk-size 8`.
The 8-frame setting adapts Lingbot's 9-frame first block and 12-frame steady
blocks. Add `--no-postprocess-compile` for development smoke tests.

See the shared [Cam2V README](../../../../apps/cam2v/README.md) for controls,
application arguments, defaults, and tests. See the
[Lingbot integration README](../../README.md) for model details.
