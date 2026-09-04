<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Lingbot World

Lingbot World is a streaming camera-controlled image-to-video model integration.
Its public model variants are `StreamInferencePipelineConfig` literals in
`lingbot.config`; the reusable Cam2V application owns interactive I/O.

## Pipeline configurations

- `PIPELINE_LINGBOT_WORLD_FAST`
- `PIPELINE_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3`
- `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST`
- `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST_TAEHV_WINDOW15_SINK3`

## Install

```bash
uv sync --package flashdreams-lingbot --inexact
```

Checkpoints download from Hugging Face on first use. Export `HF_TOKEN` when the
selected repository requires authentication.

## Cam2V application

```bash
uv sync --package flashdreams-lingbot --inexact
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

Shared [`apps/cam2v`](../../apps/cam2v/README.md) documentation lists controls,
application arguments, and development commands.

## Browser UI

This application serves its own page, so the run above offers two clients:

- `http://<host>:8089/` — the runtime's minimal viewer, the same one every v2
  application gets.
- `http://<host>:8089/request_session` — the Lingbot scene UI: a preset
  picker, a first-frame upload, a prompt box, and buttons for the scene's text
  events.

The page lives in [`apps/cam2v/web/`](apps/cam2v/web) and is served because
`LingbotCam2VApplication` implements `IWebUiProvider`
(`flashdreams.api_v2.web_ui`). That protocol is also what adds
`/api/session/initial_scene`, `/api/session/first_frame`, and
`POST /api/session/input`; an application without it gets none of them.

Useful URL parameters:

| Parameter | Effect |
| --- | --- |
| `?manual` | Do not auto-connect. Only one session runs per process, so a stray tab can otherwise claim it before the tab you meant to use. |
| `?preset=<slug>` | Open straight to a preset, slug being its lowercased name with spaces as hyphens, e.g. `?preset=water-blaster`. |

Digit keys `1`-`9` jump to the matching preset while no text field has focus.
`C` clears the active event; `R` restarts the scene, discarding the rollout's
accumulated state so it re-seeds from its first frame — which is how to
recover after prompt swaps have left the world looking like a blend of
scenes. The current prompt is re-applied to the fresh rollout.

### Scenes and text events

A scene is a prompt, a first frame, and a catalog of text events. Triggering an
event swaps the rollout's cross-attention text context in place, so the scene
changes without restarting the session; clearing it restores the scene's base
prompt. The reserved `user_prompt` event id carries free-form text typed into
the page instead of a catalog entry.

Catalog rules live in [`apps/cam2v/scene.py`](apps/cam2v/scene.py): at most 20
events, 64-character labels, 1000-character event prompts, and ids drawn from
letters, numbers, `_`, `.`, `:` and `-`. The built-in presets are in
[`apps/cam2v/web/scene_presets.json`](apps/cam2v/web/scene_presets.json).

Changing the prompt — including by picking another preset — swaps the text
context the same way, so the scene changes while the rollout keeps running. The
first frame is the exception: a rollout cannot replace the frame it was
initialized from, so picking a preset in the browser changes its events and
wording while the world still looks like whatever the session started on. To
start *in* a preset, name it when launching: `PRESET=noir-alley-combat bash
run.sh`.

## Scripts

For boxes without uv, [`setup.sh`](setup.sh) builds a plain `venv` with pip and
installs `flashdreams`, `apps/cam2v`, and this package as editable:

```bash
bash setup.sh          # TORCH_INDEX=cu130 to change the CUDA wheel index
bash run.sh            # then open http://<host>:8089/request_session
```

The camera-controls overlay is off by default here, because it is composited
into the generated frames rather than drawn by the browser, and so cannot be
dismissed from the page. `--ui` turns it on for its timing readout. A UI loop
runs either way: it is the only thing that can ask the runtime for a
replacement session, which is how the page switches scenes.

`run.sh` uses `./.venv`, or an already-active `VIRTUAL_ENV`, or whatever
`flashdreams-run-v2` is on `PATH`. Arguments are forwarded to the
application. Settings are environment variables and go *before* the command
— `LIGHT=1 bash run.sh`, not `bash run.sh LIGHT=1`:

| Variable | Default | Effect |
| --- | --- | --- |
| `APP` | `cam2v-lingbot` | Application slug; any entry point in `pyproject.toml`. |
| `HOST` / `PORT` | `0.0.0.0` / `8089` | Where to serve. |
| `LIGHT` | `0` | `LIGHT=1` generates 512x288 at 12 fps for GPUs that cannot keep up at the native 832x464. |
| `WIDTH` / `HEIGHT` / `FPS` | unset | Override the size directly; wins over `LIGHT`. |
| `PRESET` | unset | Start the rollout on a built-in preset's own image and prompt, by slug (`noir-alley-combat`). Unset runs the bundled example scene. |
| `BLOCKS` | `100000` | Blocks to generate before the run ends — set high so a run lasts until you stop it. `BLOCKS=20` for a quick smoke test. The application's own default is 20, roughly fifteen seconds. |
| `VENV` | `./.venv` | Environment to activate. |

`LIGHT` reduces what the model generates, not just what is encoded: fewer
pixels cost less end to end but look worse. The page scales whatever is
generated up to the window, so a smaller stream is softer, not smaller. There
is no encoder to choose — the v2 server hands raw frames to aiortc's software
encoder.

## Programmatic pipeline access

```python
from lingbot.config import PIPELINE_LINGBOT_WORLD_FAST

pipeline = PIPELINE_LINGBOT_WORLD_FAST.setup().to("cuda").eval()
```

## Tests

```bash
uv run --no-sync pytest integrations_v2/lingbot -m ci_cpu
```
