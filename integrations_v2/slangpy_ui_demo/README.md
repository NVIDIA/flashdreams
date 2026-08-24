<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SlangPy UI demos

Three small v2 examples:

- `slangpy-ui-text-input` keeps an editable text field in UI-loop-owned state.
- `slangpy-ui-model-output` generates a three-layer RGBA result chunk and lets the UI
  select the channel composited beneath the UI.
- `slangpy-ui-invoke-async` signals a `W` press from the UI loop to the model
  loop with `invoke_async`, toggling its output between red and blue.

## Usage

```bash
uv sync --package flashdreams-slangpy-ui-demo
uv run flashdreams-run-v2 slangpy-ui-text-input --mode webrtc
uv run flashdreams-run-v2 slangpy-ui-model-output --mode webrtc
uv run flashdreams-run-v2 slangpy-ui-invoke-async --mode webrtc
```

Open the URL printed by the WebRTC window. The live renderer requires CUDA,
Vulkan/CUDA interop, and SlangPy.

The `step_ui` callback exposes SlangPy's complete `slangpy.ui` widget surface.
See the [SlangPy UI API reference](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui)
for every available widget constructor, method, property, flag, and callback.
