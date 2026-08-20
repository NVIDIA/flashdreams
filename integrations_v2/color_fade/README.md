<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Colour Fade

Smallest end-to-end application that produces a file. It holds no model: a
session emits solid frames fading from red to green over a fixed number of
seconds, then stays green. It runs the whole file path — `IApplication`,
`ISession`, `run_session`, `Mp4ClientWindow`, `Mp4OutputSink` — on CPU.

`red_screen` is the interactive counterpart: it responds to keys and is driven
against a window with a client on the other end. This one responds to nothing,
which is what makes the file it writes the same on every run.

## What it demonstrates

- The same `IApplication` and `ISession` an interactive application implements,
  driven by the same loop against a window that has no client. Nothing here
  names an output sink or the window that holds it.
- Frames are `[-1, 1]` floats, which is what FlashDreams models emit and what a
  sink assumes of a floating point result.
- A frame's colour comes from when it plays, not from which step produced it, so
  how many frames a step generates does not change the video.
- A run longer than the fade keeps generating: the caller decides when to stop,
  not the application.

## Usage

Writes two seconds of video, the first of them fading:

```python
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from color_fade import create_app

session_desc = SessionDesc(
    output_layout=VideoTensorLayout.bcthw,
    frames_per_second_for_step=30,
    video_width=854,
    video_height=480,
)

app = create_app()
app.init(["--seconds", "1", "--frames-per-step", "10"])
try:
    # Six steps of ten frames, at thirty frames a second.
    run_session(
        app.create_session(session_desc),
        Mp4ClientWindow("fade.mp4"),
        steps=6,
    )
finally:
    app.close()
```

`steps` is what ends the run: the window reports no input, so it can never
report a close either. The application is the caller's, so one that has loaded a
model can write several files. `run_session` owns only the session and the
window, closing both before it returns.

Writing an MP4 needs an `ffmpeg` executable on `PATH`, and a frame size that is
even in both directions.

## Tests

Run from the repository root. The tests that write a file are skipped when
`ffmpeg` is missing.

```bash
uv sync --package flashdreams-color-fade --package flashdreams-red-screen --package flashdreams-null-model --group test --inexact
uv run --no-sync pytest integrations_v2/color_fade -m ci_cpu -v
```

The end-to-end test writes a real 854x480 file and reads it back. It is written
at a size a player will open, so it can be watched as well as asserted on. Send
it somewhere you can find it:

```bash
uv run --no-sync pytest integrations_v2/color_fade -k mp4 --basetemp="$HOME/fade-out"
vlc "$HOME"/fade-out/*current/fade.mp4
```

Point `--basetemp` at an empty or throwaway directory, which pytest clears
before it uses it. Somewhere under your home directory rather than `/tmp`: a
sandboxed player, a snap or a flatpak, gets a private `/tmp` and cannot see
files written to the real one. The `*current` in that path is the symlink pytest
points at the newest run.

Together with the framework tests:

```bash
uv run --no-sync pytest flashdreams/test_v2 integrations_v2 -m ci_cpu -v
```

`--inexact` matters: it stops `uv` from uninstalling the other workspace members
it was not asked about, which the framework tests import.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--seconds` | `10` | How long the fade from red to green takes. |
| `--frames-per-step` | `8` | Frames one step generates. |

The frame width and height come from the `SessionDesc`, along with the rate the
frames are meant to play at, and the step count comes from the caller that drives
the session.

Output is a `[1, 3, frames_per_step, H, W]` float32 tensor in `bcthw` layout,
carrying `[-1, 1]` values.
