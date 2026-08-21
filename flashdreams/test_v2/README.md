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
- `test_mp4_client_window.py` covers the window a run writing a file is driven
  against: no input to report, one run through the loop to show every step
  reaches the file, and the measurements it records beside the file when a run
  asks for them.
- `test_mp4_output_sink.py` covers the sink that writes an MP4, reading each file
  back to check what was encoded. Its encoding tests are skipped when `ffmpeg` is
  missing from `PATH`.
- `test_client_window_factory.py` covers each way of watching a run answering for
  itself: the window its arguments ask for, the usage error when they are
  incomplete, and what it says about where the output went. The WebRTC tests are
  skipped when the serving packages are missing, which is also why a run writing
  a file does not import them.
- `test_t2v_application.py` covers the shared text-to-video application: its
  flags, the clip it says it would generate, what it refuses to generate, and
  when the model is
  loaded. Covering it here is what lets each `integrations_v2/t2v_*` package test
  only what is particular to its own model, over the stand-in pipeline
  `flashdreams.t2v_v2.testing` ships.
- `test_t2v_session.py` covers one rollout through `T2VSession`: the prompt
  encoded into a cache once, a block generated per step, and what a reset or a
  close leaves behind. Driving a rollout is the same job for every model, so it
  is covered here rather than in each integration.
- `test_t2v_model_check.py` covers `check_t2v_model_impl` failing usefully. Every
  integration runs the passing path against its own stand-in; what a check says
  about a run that fell short is only worth covering once.
- `test_cli.py` covers `flashdreams-run-v2`: finding an application, splitting
  the command line at `--`, choosing a window, describing the session to ask for,
  and running one into a real MP4 with a stand-in for a model. An application
  that describes no session of its own is run there too, since running more than
  text-to-video is the point of the command.
- `test_metrics_output_sink.py` covers the sink that records what a run
  measured, which is a file another tool reads: what a benchmark expects of it
  is checked against the reader itself in
  `flashdreams/tests/test_benchmark_harness.py`.

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
