<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Action2V application

A reusable native-v2 action-conditioned rollout with SlangPy prompt controls
and keyboard/gamepad camera input. Model-specific assets, parsing, and pipeline
wiring live in each integration's `config.py`.

## Usage

```bash
uv sync --package flashdreams-action2v-v2
uv run flashdreams-run-v2 action2v-lingbot --mode webrtc
```

Without explicit assets the selected model integration resolves an example.
Local inputs require `--image-path` and `--action-path`; integrations may also
require `--calibration-path`.

Application packages use the regular `flashdreams-run-v2 <slug> --mode <mode>` API. The SlangPy `step_ui` callback exposes the complete [SlangPy UI API](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui).
