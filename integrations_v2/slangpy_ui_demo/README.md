<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SlangPy UI demos

Two small v2 examples:

- `slangpy-ui-text-input` keeps an editable text field in UI-thread-owned state.
- `slangpy-ui-model-output` generates a three-layer RGBA result chunk and lets the UI
  select the channel composited beneath the UI.

## Usage

```bash
uv sync --package flashdreams-slangpy-ui-demo
uv run flashdreams-run-v2 slangpy-ui-text-input --mode webrtc
uv run flashdreams-run-v2 slangpy-ui-model-output --mode webrtc
```

Open the URL printed by the WebRTC window. The live renderer requires CUDA,
Vulkan/CUDA interop, and SlangPy.

The `draw_ui` callback exposes SlangPy's complete `slangpy.ui` widget surface.
See the [SlangPy UI API reference](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui)
for every available widget constructor, method, property, flag, and callback.
