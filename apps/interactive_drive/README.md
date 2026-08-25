<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Interactive Drive

A long-running native-v2 driving application with SlangPy controls and
controller support. Its model binding is supplied by an integration adapter.

## Usage

```bash
uv sync --package flashdreams-interactive-drive-v2
uv run flashdreams-run-v2 interactive-drive --mode webrtc
```

The default scene downloads from Hugging Face on first use. Pass `-- --scene scene.usdz` to use a local scene instead.

Application packages use the regular `flashdreams-run-v2 <slug> --mode <mode>` API. The SlangPy `step_ui` callback exposes the complete [SlangPy UI API](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui).

## Tests

```bash
uv run --no-sync pytest apps/interactive_drive -m ci_cpu -v
```
