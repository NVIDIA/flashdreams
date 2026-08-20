<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Colour Fade

Smallest end-to-end application that writes a file. It holds no model: a session
emits solid frames fading from red to green over a fixed number of seconds, and
finishes once it has. It runs the whole file path — `IApplication`, `ISession`,
`ApplicationRunner`, `run_session`, `Mp4ClientWindow`, `Mp4OutputSink`, on CPU.

`red_screen` is the interactive counterpart. This one responds to nothing, which
is what makes the file it writes the same on every run.

Writing an MP4 needs an `ffmpeg` executable on `PATH`, and a frame size that is
even in both directions.

## Tests

Run from the repository root. The tests that write a file are skipped when
`ffmpeg` is missing.

```bash
uv sync --package flashdreams-color-fade --package flashdreams-red-screen --package flashdreams-null-model --group test --inexact
uv run --no-sync pytest integrations_v2/color_fade -m ci_cpu -v
```

Together with the framework tests:

```bash
uv run --no-sync pytest flashdreams/test_v2 integrations_v2 -m ci_cpu -v
```

`--inexact` matters: it stops `uv` from uninstalling the other workspace members
it was not asked about, which the framework tests import.

The end-to-end test writes a real 854x480 file, a size a player will open, so it
can be watched as well as asserted on:

```bash
uv run --no-sync pytest integrations_v2/color_fade -k mp4 --basetemp="$HOME/fade-out"
vlc "$HOME"/fade-out/*current/fade.mp4
```

Point `--basetemp` at a throwaway directory under your home rather than `/tmp`:
pytest clears it before using it, and a sandboxed player gets a private `/tmp`
and cannot see files in the real one. `*current` is the symlink pytest points at
the newest run.

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--seconds` | `10` | How long the fade from red to green takes. |
| `--frames-per-step` | `8` | Frames one step generates. A frame's colour comes from when it plays, so this does not change the video. |

The frame width and height come from the `SessionDesc`, along with the rate the
frames are meant to play at. How long the run lasts comes from `--seconds`: the
session reports itself finished once it has generated that much fade, so the
caller driving it passes no step count.

Output is a `[1, 3, frames_per_step, H, W]` float32 tensor in `bcthw` layout,
carrying `[-1, 1]` values.
