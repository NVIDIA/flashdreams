<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# ImGui UI demos

Small ImGui applications for the FlashDreams v2 loop runtime:

- `imgui-ui-text-input` renders an editable text field over model output.
- `imgui-ui-query-string` sets its background from a `?(r,g,b)` browser URL.

## Run it

Setup:
```bash
uv sync --package flashdreams-imgui-ui-demo --inexact
```

Run the text-input application:
```bash
uv run --no-sync flashdreams-run-v2 imgui-ui-text-input --mode native-window
```

Run the query-string application:
```bash
uv run --no-sync flashdreams-run-v2 imgui-ui-query-string \
  --mode webrtc --host 127.0.0.1 --port 8080

# Connect to the WebRTC server with a query string of format `?(r,g,b)` to set the background color. Example: `http://127.0.0.1:8080/?(255,128,0)`
```

## Tests

```bash
uv sync --package flashdreams-imgui-ui-demo --inexact
uv sync --group test --inexact
uv run --no-sync pytest integrations_v2/imgui_ui_demo -m ci_cpu -v
```
