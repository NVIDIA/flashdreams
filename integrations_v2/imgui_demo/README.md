<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ImGui demos

Two small v2 examples:

- `imgui-text-input` keeps an editable text field in UI-thread-owned state.
- `imgui-model-output` generates a three-layer RGBA result chunk and lets the UI
  select the channel composited beneath the UI.

## Usage

```bash
uv sync --package flashdreams-imgui-demo --inexact
uv run --no-sync flashdreams-run-v2 imgui-text-input --mode webrtc
uv run --no-sync flashdreams-run-v2 imgui-model-output --mode webrtc
```

Open the URL printed by the WebRTC window. The live renderer requires CUDA,
Vulkan/CUDA interop, and SlangPy.
