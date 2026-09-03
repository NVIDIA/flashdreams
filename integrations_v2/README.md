<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Applications built on the v2 API. Each directory is a standalone package that
depends on `flashdreams` and holds no framework code of its own.

This is the guide to writing one. If the model generates video from a prompt,
read [`apps/t2v`](../apps/t2v/README.md)
instead, that path is a subclass and a `pyproject.toml`, and most of what
follows is already done for you.

## What is here

- `color_fade` — the smallest integration that writes a file. No model, CPU
  only, and the worked example for the rest of this guide.
- `red_screen` — the smallest interactive one, streaming to a browser.
- `slangpy_ui_demo` — three applications that draw widgets over model output,
  and the reference for writing a UI loop.
- `lingbot` — the Lingbot World model and its `cam2v-lingbot` binding to the
  shared interactive camera-to-video application.
- `waypoint` — the Waypoint model binding to the shared `apps/action2v`
  application.
- `hy_worldplay` — the HY-WorldPlay model and its `cam2v-hy-worldplay` binding,
  including live PRoPE/action camera-history adaptation.
- `waypoint` — the interactive Waypoint 1.5 image-established application
  with deterministic control replay and live keyboard/mouse input.
- `self_forcing`, `causal_forcing`, `fastvideo_causal_wan22`, `wan21`,
  `cosmos_predict2`, and `wan22` — model implementations with T2V adapters
  over the reusable `apps/t2v` package.
- `flashvsr` — streaming video super-resolution bound to the shared
  `apps/v2v` video-to-video application.
- `null_model` — not an application. A v1 pipeline the framework tests use as a
  fixture.

## The layout

```text
integrations_v2/<model>/
  pyproject.toml
  README.md
  __init__.py
  config.py            # model's unique pipeline config or config wrapper
  impl/                # all model-specific implementation
  tests/               # model-specific tests, when needed
  apps/
    <demo>/
      __init__.py
      adapter.py       # create_app() -> IApplication
      README.md        # launch instructions only
```

The workspace glob in the root `pyproject.toml` picks up any directory here
that has a `pyproject.toml`. Packages with multiple demos mirror the
`integrations_v2/omnidreams/apps/` layout.

V2 integrations expose pipeline-config literals and application entry points
directly. Each model keeps its single `config.py` at the integration root,
either as a `StreamInferencePipelineConfig` literal or a model-specific config
wrapper. Keep all other model-specific implementation under
`impl/`; no other implementation modules belong at the integration root. Keep
model-specific tests under `tests/`, not in the demo folder. Do not add a `runner.py` or
`flashdreams.runner_configs` entry point; the model adapter supplies its
application defaults from its pipeline config.

## The package metadata

The parts that matter, taking
[`color_fade`](color_fade/pyproject.toml) as the shape and
[`self_forcing`](self_forcing/pyproject.toml) for the entry point:

```toml
[project]
name = "flashdreams-color-fade"
requires-python = ">=3.10"
dependencies = ["flashdreams"]

[tool.uv.sources]
flashdreams = { workspace = true }

[tool.setuptools.packages.find]
include = ["color_fade*"]

# color_fade registers no entry point and is found by the module fallback below.
# Anything real should register one, the way self_forcing does:
[project.entry-points."flashdreams.applications_v2"]
"t2v-self-forcing-wan2.1-t2v-1.3b" = "self_forcing.apps.t2v.adapter:create_app"
"t2v-self-forcing-wan2.1-t2v-1.3b-taehv" = "self_forcing.apps.t2v.adapter:create_app_taehv"

[tool.setuptools.package-dir]
self_forcing = "."
```

The default entry point is ``<demo-slug>-<model-slug>`` and uses
``create_app``. Additional compatible configurations may append a descriptive
``-<suffix>`` (for example ``-fast``) and use the matching
``create_app_<suffix>`` factory in the same adapter. Multiword entry-point
suffixes stay hyphenated while the Python factory uses underscores.

An integration depends on the framework and never the reverse. `tool.uv.sources`
resolves `flashdreams` from this repository while developing; a published
integration would resolve a released version instead. A real model adds its
model package alongside, also as a workspace source. An integration that streams
to a browser depends on `flashdreams[serving]`, and `slangpy_ui_demo` adds
`local-window` on top of that for its renderer.

## Being found

`flashdreams-run-v2` takes a slug and resolves it two ways, in order. A
registered `flashdreams.applications_v2` entry point wins. Failing that, the slug
is read as a module name with hyphens turned into underscores, and that module
must expose `create_app`.

The fallback is why `color_fade` and `red_screen` run while registering nothing:
the slug `color-fade` finds the `color_fade` package, whose `__init__.py`
re-exports `create_app`. Every t2v integration and `slangpy_ui_demo` register
properly instead. Do the same for anything real, so the slug is listed by
`flashdreams-run-v2 --help` rather than having to be known already.

Every `create_app` or `create_app_<suffix>` factory takes no arguments and
returns an uninitialized `IApplication`. It must not load anything; that is
what `init` is for.

## What to implement

Three classes, and a fourth only if you want custom UI. The full versions are in
[`color_fade/color_fade/app.py`](color_fade/color_fade/app.py).

**The application** parses its own arguments and holds whatever its sessions
share. Arguments arrive as a list, after `--` on the command line, so it owns a
parser of its own and may reuse names the runtime also uses:

```python
class ColorFadeApplication(IApplication):
    def init(self, commandline_args: Sequence[str]) -> None:
        parser = argparse.ArgumentParser(prog="color-fade")
        parser.add_argument("--seconds", type=float, default=10.0)
        args = parser.parse_args(list(commandline_args))
        self._config = ColorFadeConfig(seconds=args.seconds)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        return ColorFadeSession(self._config, session_desc)
```

Validate arguments here rather than at the first step. `color_fade` rejects a
fade of `nan` seconds because every frame would then be `nan`, and `nan` frames
reach an output sink as a picture rather than as an error.

An application may also implement `session_desc` to say what it would generate
before it has loaded anything, which is how a real model reports the frame size
its checkpoint was trained for. Returning `None`, the default, means it
generates whatever it is asked for.

**The session** is one run. It checks the description it was handed, registers a
model loop in `init`, and reports the description back:

```python
class ColorFadeSession(ISession):
    def __init__(self, config: ColorFadeConfig, session_desc: SessionDesc) -> None:
        if session_desc.output_layout is not VideoTensorLayout.bcthw:
            raise ValueError("Colour fade only produces bcthw output.")
        self._session_desc = session_desc

    def init(self) -> None:
        self.register_model_loop(ColorFadeModelLoop, state=...)

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc
```

Reject a description you cannot honour, in the constructor or in
`create_session`. A session that quietly ignores the layout it was asked for
produces tensors an output sink then refuses, much further along.

**The model loop** generates. It returns a list of channels, and it decides when
the run is over:

```python
class ColorFadeModelLoop(IModelLoop[ColorFadeModelState]):
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        del events
        self.state.steps_generated += 1
        return [StepResult(step_index=step_index, output=..., frame_count=8,
                           output_layout=self.state.session_desc.output_layout)]

    def is_finished(self) -> bool:
        return self.state.steps_generated >= self.state.total_steps

    def reset(self) -> None:
        self.state.steps_generated = 0
```

Three things catch people out here. A model loop must return a `list`, even of
one channel. `is_finished` is what ends a run writing to a file, since an MP4
window never sends a close event. And `reset` raises by default, so implement it
even when the body is one line — a browser client can ask for one at any time.

Register no UI loop unless you need one. The default composites every model
channel into one frame, which is what most demos need (Refer to `slangpy_ui_demo` for more complex behavior via `SlangPyUILoop`; Refer to `interactive_drive` for more complex behavior via `ImGuiUILoop`).

## Running it

```bash
uv sync --package flashdreams-color-fade --inexact
uv run --no-sync flashdreams-run-v2 color-fade --output-path fade.mp4 -- --seconds 4
```

Arguments before `--` belong to the runtime, after it to the application, so
`flashdreams-run-v2 color-fade -- --help` describes the application. Writing an
MP4 needs `ffmpeg` on `PATH` and a frame size even in both directions.

To stream to a browser instead:

```bash
uv run --no-sync flashdreams-run-v2 red-screen --mode webrtc
```

Driving it from Python takes the same two objects the command line builds:

```python
ApplicationRunner(create_app(), Mp4ClientWindow(path)).run(session_desc, args)
```

## Testing it

Every test carries a `ci_cpu`, `ci_gpu` or `manual` marker, enforced by a pytest
plugin, so a file usually sets `pytestmark` once:

```python
pytestmark = pytest.mark.ci_cpu
```

An integration with no model should be entirely `ci_cpu`, testing the session
directly by stepping its model loop, and end to end through `ApplicationRunner`
against a real `Mp4ClientWindow`. Skip the file tests when `ffmpeg` is missing
rather than failing them. For interactive integrations, drive input with a
scripted `IClientWindow`, as `red_screen` does.

```bash
uv sync --package flashdreams-color-fade --group test --inexact
uv run --no-sync pytest integrations_v2/color_fade -m ci_cpu -v
```

`--inexact` matters: without it `uv` uninstalls the workspace members it was not
asked about, which the framework tests import.

Point an end-to-end test at a size a player will open, and write it somewhere you
can watch it:

```bash
uv run --no-sync pytest integrations_v2/color_fade -k mp4 --basetemp="$HOME/fade-out"
vlc "$HOME"/fade-out/*current/fade.mp4
```

Use a throwaway directory under your home rather than `/tmp`: pytest clears the
base temp before using it, and a sandboxed player gets a private `/tmp` and
cannot see files in the real one.

A real model needs a second file. Keep the CPU tests on a stand-in pipeline and
put the run that loads a checkpoint behind `ci_gpu` and an environment variable,
so a GPU runner opts in explicitly. `t2v.testing` provides both
halves for text-to-video models.

## Where to go next

- [Architecture](../ARCHITECTURE.md) - how an application, a session and the
  runtime fit together, and the two threads your loops run on.
- [v2 API protocols](../flashdreams/flashdreams/api_v2/README.md) - the contracts
  in detail.
- [Runtime](../flashdreams/flashdreams/runtime_v2/README.md) - the buffering
  between those threads, and the command line that starts them.
- [`apps/t2v`](../apps/t2v/README.md) - adding a
  text-to-video model.
