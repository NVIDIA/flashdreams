<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SlangPy UI demos

The reference for writing a UI loop. Everywhere else here registers no UI loop and
lets the runtime blit the model output; these three register one and draw widgets
over it.

- `slangpy-ui-text-input` keeps an editable text field in UI-loop-owned state.
- `slangpy-ui-model-output` generates a three-layer RGBA result chunk and lets the
  UI select which channel is composited beneath it. The one to read for how a UI
  loop reaches model output.
- `slangpy-ui-invoke-async` signals a `W` press from the UI loop to the model
  loop with `invoke_async`, toggling its output between red and blue. The one to
  read for crossing between the two threads.

## Run them

```bash
uv sync --package flashdreams-slangpy-ui-demo --inexact
uv run --no-sync flashdreams-run-v2 slangpy-ui-text-input --mode webrtc
uv run --no-sync flashdreams-run-v2 slangpy-ui-model-output --mode webrtc
uv run --no-sync flashdreams-run-v2 slangpy-ui-invoke-async --mode webrtc
uv run --no-sync flashdreams-run-v2 slangpy-ui-model-output --mode native-window
```

Open the URL the WebRTC window prints. The live renderer needs CUDA,
Vulkan/CUDA interop, and SlangPy, which is why these are the only integrations
here that will not run on a CPU. Native-window mode presents the result directly
in a GLFW window and keeps GPU-resident tensors off the CPU.

## Writing one

Subclass `SlangPyUILoop` from `flashdreams.runtime_v2.slangpy_ui_loop` and
implement `step_ui(ui, step_index, events)` rather than `step`. The `ui` argument
exposes `ui.screen`, the root `slangpy.ui.Screen` that receives top-level
widgets, and every public type from `slangpy.ui` alongside it.

FlashDreams delegates those names straight through rather than wrapping them, so
the [SlangPy UI API reference](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui)
is the source of truth for every widget constructor, method, property, flag and
callback.

Model frames come from `presented_model_frame` and `presented_model_frames`, not
from the model loop directly, see [the architecture](../../ARCHITECTURE.md) for
why the two loops do not share memory.

## Tests

```bash
uv sync --package flashdreams-slangpy-ui-demo --inexact
uv sync --group test --inexact
uv run --no-sync pytest integrations_v2/slangpy_ui_demo -m ci_cpu -v
```

These cover the loops without a renderer, so they run on a CPU even though the
applications themselves do not.
