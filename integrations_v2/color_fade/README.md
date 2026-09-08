<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Colour Fade

Smallest end-to-end application that writes a file, and the worked example in the
[integration guide](../README.md). It holds no model: a session emits solid frames
fading from red to green over a fixed number of seconds, and finishes once it
has. That exercises the whole file path — `IApplication`, `ISession`,
`ApplicationRunner`, `run_session`, `Mp4ClientWindow`, `Mp4OutputSink` — on CPU.

`red_screen` is the interactive counterpart. This one responds to nothing, which
is what makes the file it writes the same on every run.

## Generate a fade

```bash
uv sync --package flashdreams-color-fade --inexact
uv run --no-sync flashdreams-run-v2 color-fade --output-path fade.mp4 \
    -- --seconds 4
```

Writing an MP4 needs an `ffmpeg` executable on `PATH`, and a frame size that is
even in both directions. The run ends on its own: the session knows how long its
fade is, so nothing has to pass a step count.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--seconds` | `10` | How long the fade from red to green takes. |
| `--frames-per-step` | `8` | Frames one step generates. A frame's colour comes from when it plays, so this does not change the video. |

Frame width, height and playback rate come from the `SessionDesc`, so they are
set with `--pixel-width`, `--pixel-height` and `--fps` before the `--`.

Output is a `[1, 3, frames_per_step, H, W]` float32 tensor in `bcthw` layout,
carrying `[-1, 1]` values. Any other layout is refused.

## Tests

```bash
uv sync --package flashdreams-color-fade --group test --inexact
uv run --no-sync pytest integrations_v2/color_fade -m ci_cpu -v
```

The tests that write a file are skipped when `ffmpeg` is missing. The end-to-end
one writes a real 854x480 file, a size a player will open, so it can be watched
as well as asserted on:

```bash
uv run --no-sync pytest integrations_v2/color_fade -k mp4 --basetemp="$HOME/fade-out"
vlc "$HOME"/fade-out/*current/fade.mp4
```

See the [integration guide](../README.md) for what `--inexact` and `--basetemp`
are doing there.
