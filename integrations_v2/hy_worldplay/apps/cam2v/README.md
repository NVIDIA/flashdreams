<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HY-WorldPlay Cam2V

Install the HY-WorldPlay integration and launch its `cam2v-hy-worldplay`
application:

```bash
uv sync --package flashdreams-hy-worldplay --inexact
uv run --no-sync flashdreams-run-v2 cam2v-hy-worldplay \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

See the shared [Cam2V README](../../../../apps/cam2v/README.md) for controls,
application arguments, defaults, and tests. The application uses
`PIPELINE_HY_WORLDPLAY_WAN_I2V_5B`; see the
[HY-WorldPlay integration README](../../README.md) for model details.
