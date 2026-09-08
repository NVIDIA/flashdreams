<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ImGui UI demo

An editable ImGui text field rendered over model output by the FlashDreams v2
loop runtime.

## Run it

```bash
uv sync --package flashdreams-imgui-ui-demo --inexact
uv run --no-sync flashdreams-run-v2 imgui-ui-text-input --mode native-window
```

The live renderer requires CUDA, Vulkan/CUDA interop, and SlangPy.

## Tests

```bash
uv sync --package flashdreams-imgui-ui-demo --inexact
uv sync --group test --inexact
uv run --no-sync pytest integrations_v2/imgui_ui_demo -m ci_cpu -v
```
