<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SlangPy UI demos

Three examples cover retained text input, model-output channel selection, and cross-loop `invoke_async` messaging.

## Usage

```bash
uv sync --package flashdreams-slangpy-ui-demo
uv run flashdreams-run-v2 slangpy-ui-text-input --mode webrtc
```

Also try `slangpy-ui-model-output` and `slangpy-ui-invoke-async`. Live rendering requires CUDA, Vulkan/CUDA interop, and SlangPy.

Application packages use the regular `flashdreams-run-v2 <slug> --mode <mode>` API. The SlangPy `step_ui` callback exposes the complete [SlangPy UI API](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui).

## Tests

```bash
uv run --no-sync pytest apps/slangpy_ui_demo -m ci_cpu -v
```
