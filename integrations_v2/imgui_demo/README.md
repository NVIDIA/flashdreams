<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ImGui demos

Two small v2 threaded-runtime examples:

- `imgui-text-input` keeps an editable text field in UI-thread-owned state.
- `imgui-model-output` generates a three-layer RGBA result chunk and lets the UI
  select a channel for `draw_presented_model_frame` to draw.
- `imgui-model-output -- --no-ui` omits UI registration so the session-provided
  default composites model channels directly into its one presentation frame.

## Usage

```bash
uv sync --package flashdreams-imgui-demo --inexact
uv run --no-sync flashdreams-run-v2 imgui-text-input --mode webrtc
uv run --no-sync flashdreams-run-v2 imgui-model-output --mode webrtc
uv run --no-sync flashdreams-run-v2 imgui-model-output --mode webrtc -- --no-ui
```

Open the URL printed by the WebRTC window. The live renderer requires CUDA,
Vulkan/CUDA interop, SlangPy, and `imgui-bundle`.
