<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Colour fade

A deterministic red-to-green v2 application whose model loop owns generation state.

## Usage

```bash
uv sync --package flashdreams-color-fade
uv run flashdreams-run-v2 color-fade --mode mp4 --output-path fade.mp4
```

Application arguments follow `--`, for example `-- --seconds 5 --frames-per-step 8`. The same slug also supports `--mode webrtc`.

Application packages use the regular `flashdreams-run-v2 <slug> --mode <mode>` API.

## Tests

```bash
uv run --no-sync pytest apps/color_fade -m ci_cpu -v
```
