<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Red Screen

Smallest end-to-end application on the v2 API. It holds no model: a session emits
red frames controlled by activation and intensity keys. It runs the whole path —
`IApplication`, `ISession`, `run_session`, `IClientWindow` — on CPU.

## What it demonstrates

- An application module implements `IApplication` and `ISession` and nothing
  else. This one never names `IClientWindow`, `InputSource` or `OutputSink`.
- A session is created from a `SessionDesc` with no client window involved, so it
  works before any client connects. It reports what it resolved to in
  `ISession.session_desc`, which `run_session` hands to `IClientWindow.open`.
- An application can reject a description it cannot honour: this one raises on
  any layout other than `bcthw`.
- The runner drives the loop and hands the session each `step_index`, so the
  session keeps no step count of its own. The `steps=4` below is what bounds this
  example; an interactive window would pass `steps=None` and end the run by
  reporting a close event instead.
- Keyboard events are edges, not levels. A key stays held across steps that carry
  no events for it, so a key-down at step 0 keeps the screen red until the
  matching key-up arrives.
- `reset()` is implemented, so one session can run again.

## Usage

Start the WebRTC server and open the printed URL:

```bash
uv run red-screen-webrtc
```

Hold `r` in the browser to turn the generated video red, or click **Activate**
to toggle the same keyboard event. Press `w` to increase the red intensity by
0.1 and `s` to decrease it by 0.1. Runtime options configure the browser session:

```bash
uv run red-screen-webrtc --mode webrtc --host 0.0.0.0 --port 8080 --width 1280 --height 720 --fps 30
```

Arguments after `--` belong to the application:

```bash
uv run red-screen-webrtc -- --key x
```

The same application can be driven directly without WebRTC:

```python
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from red_screen import create_app

app = create_app()
app.init([])
session = app.create_session(
    SessionDesc(
        output_layout=VideoTensorLayout.bcthw,
        video_width=16,
        video_height=16,
    )
)
run_session(session, my_client_window, steps=4)
app.close()
```

## Tests

Run from the repository root.

```bash
uv sync --package flashdreams-red-screen --package flashdreams-null-model --package flashdreams-color-fade --group test --inexact
uv run --no-sync pytest integrations_v2/red_screen -m ci_cpu -v
```

Together with the framework tests:

```bash
uv run --no-sync pytest flashdreams/test_v2 integrations_v2 -m ci_cpu -v
```

`--inexact` matters: it stops `uv` from uninstalling the other workspace members
it was not asked about, which the framework tests import.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--key` | `r` | Key whose held state selects red over black. |

The frame width and height come from the `SessionDesc`, and the step count from
the caller that drives the session. Neither is a command-line argument.

Output is a `[1, 3, 1, H, W]` float32 tensor in `bcthw` layout, carrying the
`[-1, 1]` values a client window expects: red is `1.0` on channel 0 and `-1.0`
on the others, and black is `-1.0` everywhere.
