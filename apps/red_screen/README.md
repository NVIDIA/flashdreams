<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Red screen

An interactive v2 application that consumes keyboard edges on the model loop and emits black or red frames.

## Usage

```bash
uv sync --package flashdreams-red-screen
uv run flashdreams-run-v2 red-screen --mode webrtc
```

Open the printed URL. Use `--mode mp4 --output-path red.mp4` with the same slug; client-window selection stays in the regular runtime.

Application packages use the regular `flashdreams-run-v2 <slug> --mode <mode>` API.

## Tests

```bash
uv run --no-sync pytest apps/red_screen -m ci_cpu -v
```
