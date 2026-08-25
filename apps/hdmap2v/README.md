<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HDMap2V application

A native v2 scene-driving rollout with SlangPy prompt/view controls and keyboard, gamepad, and wheel input.

## Usage

```bash
uv sync --package flashdreams-hdmap2v-v2
uv run flashdreams-run-v2 hdmap2v --mode webrtc -- --scene scene.usdz
```

Use `--backend raster` for conditioning output or `--backend world_model` for
the model integration selected by the adapter. Scene, simulation, and backend
work stay on the model loop.

Application packages use the regular `flashdreams-run-v2 <slug> --mode <mode>` API. The SlangPy `step_ui` callback exposes the complete [SlangPy UI API](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui).
