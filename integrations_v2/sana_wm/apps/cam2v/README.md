<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SANA-WM Cam2V

```bash
uv sync --package flashdreams-sana-wm --inexact
uv run --no-sync flashdreams-run-v2 cam2v-sana-wm-streaming \
  --mode webrtc --host 0.0.0.0 --port 8089 -- \
  --example-data
```

The app uses the streaming SANA-WM checkpoint at 1280x704 and emits ten
24-frame blocks by default. Pass `--total-blocks` to change the finite
rollout length. `--example-data` downloads the official `demo_0.png` and paired
prompt to the FlashDreams cache. Explicit `--image-path` and prompt arguments
override them.

Shared [`apps/cam2v`](../../../../apps/cam2v/README.md) documentation lists
controls, output modes, and common application arguments.
