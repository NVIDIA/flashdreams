<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# V2 framework tests

CPU-only tests for the v2 protocols themselves:

- `test_client_window.py` drives the I/O protocols against the deterministic NULL
  model integration.
- `test_session_runner.py` covers the `run_session` loop with fake session and
  window implementations, so it depends on no integration at all. It asserts the
  orderings and thread ownership the two-thread loop guarantees, not a particular
  interleaving.
- `test_batch_runner.py` covers the `run_batch` loop the same way, with a fake
  session and sink, including what it closes when a run fails part way through.
- `test_mp4_output_sink.py` covers the sink that writes an MP4, reading each file
  back to check what was encoded. Its encoding tests are skipped when `ffmpeg` is
  missing from `PATH`.

Application behaviour is tested by the application that owns it — see
`integrations_v2/red_screen/red_screen/tests/` and
`integrations_v2/color_fade/color_fade/tests/`.

Run commands from the repository root.

## Set up the test environment

```bash
uv sync --package flashdreams-color-fade --package flashdreams-red-screen --package flashdreams-null-model --group test --inexact
```

`test_client_window.py` imports the NULL model integration, and naming every
integration leaves the environment ready for their tests too. `--inexact`
matters: without it, `uv` makes the environment exact for the packages it was
given and uninstalls the rest. `pytest` comes from the `test` group; do not use
`--extra dev`, which pulls `transformer-engine` and compiles CUDA extensions from
source.

## Run the tests

```bash
uv run --no-sync pytest flashdreams/test_v2 -m ci_cpu -v
```

A single test:

```bash
uv run --no-sync pytest flashdreams/test_v2/test_session_runner.py -v
```

`--no-sync` keeps the run from re-resolving the environment.

The tests are marked `ci_cpu`; they need no GPU and no model checkpoint.
