<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashVSR Upsample Video

This adapter binds the shared
[Upsample Video](../../../../apps/upsample_video/README.md) application to three
FlashVSR configurations.

| Entry-point slug | Configuration |
| --- | --- |
| <code>upsample-video-flashvsr-v1.1-sparse-ratio-2.0</code> | Stable sparse attention. |
| <code>upsample-video-flashvsr-v1.1-sparse-ratio-1.5</code> | Faster sparse attention. |
| <code>upsample-video-flashvsr-v1.1-full-attn</code> | Dense full attention. |

Launch any slug with the v2 application runner:

```bash
uv sync --package flashdreams-flashvsr --inexact
uv run --no-sync flashdreams-run-v2 upsample-video-flashvsr-v1.1-sparse-ratio-2.0 --output-path big-buck-bunny-upscaled.mp4 -- --max-chunks 4
```

See the shared app README for controls, application arguments, presentation
modes, and its CPU test command.
