---
title: 'Testing'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Pytest collection is configured in the root `pyproject.toml`. There is
no `testpaths` setting, so `pytest` from the repo root walks the tree:

- Files named `test_*.py`. Pytest's default also collects `*_test.py`;
  don't use that name here.
- Classes named `Test*` (pytest default). Some modules use classes,
  some only module-level functions.
- Functions and methods named `test_*` (pytest default), including
  `async def test_*`.

Same test filename in different folders is fine (several
`test_application.py` files exist). Root `pyproject.toml` already sets
`--import-mode=importlib` so pytest keeps them separate. Do not remove
that flag.

A `test_*.py` next to the code it validates is collected with no CI
change. The root `pyproject.toml` skips `parity_check`,
`parity_check_v2`, `baseline_fastvideo`, and `baseline_lightx2v` via
`norecursedirs`.

Every test function must be marked with exactly one **CI tier marker**.
A pytest plugin (`flashdreams._pytest_plugins.marker_enforcement`)
enforces this at collection time -- tests without a marker are
rejected, and tests with both `ci_cpu` and `ci_gpu` are rejected.

| Marker | When to use | CI runner |
|--------|-------------|-----------|
| `@pytest.mark.ci_cpu` | Pure CPU logic, no GPU or `libGL` needed | CPU runner |
| `@pytest.mark.ci_gpu` | Needs CUDA, `libGL`, or transitive `cv2` | GPU runner (RTX Pro 6000) |
| `@pytest.mark.manual` | Heavy (OOM risk), flaky, needs credentials or large downloads | Not run in CI |

Use a module-level `pytestmark` when every test in a file shares the
same marker:

```python
import pytest

pytestmark = pytest.mark.ci_cpu
```

Use per-function markers when tests in the same file have different
tiers:

```python
@pytest.mark.ci_cpu
def test_basic_math(): ...

@pytest.mark.ci_gpu
def test_cudagraph_path(): ...
```

Running tests locally:

```bash
uv run pytest -m ci_cpu          # CPU-safe tests only
uv run pytest -m ci_gpu          # GPU tests only (needs CUDA)
uv run pytest -m "not manual"    # everything that runs in CI
uv run pytest                    # all tests including manual
```

For local and Docker test-runner scripts, see
[tests/README.md](https://github.com/NVIDIA/flashdreams/blob/main/tests/README.md).

## Synthetic test demos

| Demo | Purpose | Instructions |
| --- | --- | --- |
| NULL Model | Provides a deterministic arithmetic pipeline for CPU testing without checkpoints. | [Read instructions](#null-model) |
| Colour Fade | Writes a bounded red-to-green video to exercise the file-output runtime. | [Read instructions](#colour-fade) |
| Red Screen | Exercises interactive keyboard input by changing a synthetic red screen. | [Read instructions](#red-screen) |
| ImGui UI demos | Exercises ImGui text input, browser query strings, and window resizing. | [Read instructions](#imgui-ui-demos) |
| SlangPy UI demos | Exercises SlangPy widgets, model-output selection, and cross-thread signaling. | [Read instructions](#slangpy-ui-demos) |

### NULL Model

A simple pipeline wth an encoder, transformer, scheduler and decoder, whose output
is arithmetic rather than video, so it needs no checkpoint and runs on a CPU. It
was written alongside the v2 protocols as the pipeline they were developed against,
something whole that could be driven end to end while they were still changing,
which is why it sits here rather than under `integrations/`.

`flashdreams/test_v2/test_client_window.py` uses the arithmetic pipeline as a
CPU test fixture. The synthetic runtime demos are defined under `apps/`; they
generate their own frames without `NULL_MODEL_CONFIG` or checkpoints. Their
`create_app` entry points live under `integrations_v2/null_model/null_model/apps/`.

| Demo | Purpose | App path (relative to repo root) |
| --- | --- | --- |
| [Colour Fade](#colour-fade) | Bounded synthetic video output. | `apps/color_fade/` |
| [Red Screen](#red-screen) | Keyboard input and interactive output. | `apps/red_screen/` |
| [ImGui UI](#imgui-ui-demos) | Text input, query strings, and resizing. | `apps/imgui_ui_demo/` |
| [SlangPy UI](#slangpy-ui-demos) | Widgets, channel selection, and cross-thread signaling. | `apps/slangpy_ui_demo/` |

```bash
uv sync --package flashdreams-null-model --extra demos --inexact
uv run --no-sync flashdreams-run-v2 red-screen --mode webrtc
```

The existing demo slugs are registered by this package. The pipeline itself is
not an application and is not connected to these demos. The remainder of this
README describes the arithmetic pipeline.

#### Observable contract

| Property | Value |
| --- | --- |
| Input | Tensor with shape `[1, 1]` |
| Output shape | Tensor with shape `[1, 3, 1, 1, 1]` |
| Output value | `Input + cache.autoregressive_index` |
| Output layout | `VideoTensorLayout.bcthw` |

#### Files

| File | What it does |
| --- | --- |
| `config.py` | Defines the null-model pipeline. |
| `encoder.py` | Adds 100 to the input, as minor obfuscation. |
| `transformer.py` | Turns the encoded input into a flow for the scheduler to denoise. |
| `decoder.py` | Subtracts the 100 back off. |

```python
NULL_MODEL_CONFIG = NullModelConfig(
    name="null-model",
    encoder=NullInputEncoderConfig(),
    diffusion_model=DiffusionModelConfig(
        transformer=NullTransformerConfig(),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=1,
            denoising_timesteps=[1000],
        ),
    ),
    decoder=NullDecoderConfig(),
)
```

#### How the pipeline is put together

##### A real integration package

[`pyproject.toml`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/pyproject.toml) declares `flashdreams-null-model` as a
workspace package depending on `flashdreams`.

##### The per-step encoder

[`encoder.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/null_model/encoder.py) defines `NullInputEncoder` as a
`StreamingEncoder`, bound in the config to `NullModelConfig.encoder`. A streaming
encoder runs on every autoregressive step, unlike
`NullModelConfig.diffusion_model.transformer.context_encoder`, which runs once at
the start of a generation.

It adds 100 to the 1x1 input tensor.

##### The transformer

[`transformer.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/null_model/transformer.py) defines, in order:

1. `latent_shape` — one batch, three channels, one frame, one pixel.
2. `initialize_autoregressive_cache()` — a cache tracking the autoregressive step
   of a continuous generation.
3. `initial_noise()` — zeros, so the scheduler has no noise to denoise beyond the
   flow `predict_flow` returns.
4. `predict_flow()` — the flow, from the encoded input and the step index the
   cache reports.

```text
# NullTransformer
target = encoded_input + cache.autoregressive_index
flow   = noisy_latent - target

# FlowMatchScheduler, sigma 1.0, one step
clean = noisy_latent - 1.0 * flow
      = noisy_latent - (noisy_latent - target)
      = target
```

Which is why one scheduler step is enough, and why every output tensor is exactly
the expected value.

##### The per-step decoder

[`decoder.py`](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/null_model/null_model/decoder.py) defines `NullDecoder`, which subtracts the
encoder's 100 back off the transformer's output.

#### Tests

```bash
uv run --no-sync pytest integrations_v2/null_model
```

### Colour Fade

Smallest end-to-end application. It holds no model: a session emits solid frames
fading from red to green over a fixed number of seconds, and finishes once it
has. That exercises the whole file path — `IApplication`, `ISession`,
`ApplicationRunner`, `run_session`, `Mp4ClientWindow`, `Mp4OutputSink` — on CPU.

`red_screen` is the interactive counterpart. This one responds to nothing, which
is what makes the file it writes the same on every run.

#### Generate a fade

```bash
uv sync --package flashdreams-null-model --extra demos --inexact
uv run --no-sync flashdreams-run-v2 color-fade --output-path fade.mp4 \
    --timeout unbound -- --seconds 4
```

### Red Screen

Smallest interactive application on the v2 API. It holds no model: a session emits
red frames whose intensity is driven by keys held in the browser. `color_fade` is
the counterpart that writes a file; this one is about input.

#### Run it

```bash
uv sync --package flashdreams-null-model --extra demos --inexact
uv run --no-sync flashdreams-run-v2 red-screen --mode webrtc
```

Open the URL it prints. Hold `r` to turn the generated video red, `w` to raise
the red intensity by 0.1, and `s` to lower it.

### ImGui UI demos

Small ImGui applications for the FlashDreams v2 loop runtime:

- `imgui-ui-text-input` renders an editable text field over model output.
- `imgui-ui-query-string` sets background from a `?(r,g,b)` browser URL.
- `imgui-ui-window-size` resizes native window or UI render target without
  resetting session.

#### Run it

Setup:
```bash
uv sync --package flashdreams-null-model --extra demos --inexact
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

Run the window-size application:
```bash
uv run --no-sync flashdreams-run-v2 imgui-ui-window-size --mode native-window
```

Schedule any number of window or UI render-target resizes by repeating paired
application arguments after ``--``:
```bash
uv run --no-sync flashdreams-run-v2 imgui-ui-window-size \
  --mode mp4 --output-path window-resize.mp4 --timeout 3 -- \
  --resize-after-ui-loops 15 --resize-to-size 320x240 \
  --resize-ui-after-ui-loops 30 --resize-to-size 800x600 \
  --resize-after-ui-loops 45 --resize-to-size 400x300
```

### SlangPy UI demos

The reference for writing a UI loop. Everywhere else here registers no UI loop and
lets the runtime blit the model output; these three register one and draw widgets
over it.

- `slangpy-ui-text-input` keeps an editable text field in UI-loop-owned state.
- `slangpy-ui-model-output` generates a three-layer RGBA result chunk and lets the
  UI select which channel is composited beneath it. The one to read for how a UI
  loop reaches model output.
- `slangpy-ui-invoke-async` signals a `W` press from the UI loop to the model
  loop with `invoke_async`, toggling its output between red and blue. The one to
  read for crossing between the two threads.

#### Run them

```bash
uv sync --package flashdreams-null-model --extra demos --inexact
uv run --no-sync flashdreams-run-v2 slangpy-ui-text-input --mode webrtc

uv run --no-sync flashdreams-run-v2 slangpy-ui-model-output --mode webrtc
uv run --no-sync flashdreams-run-v2 slangpy-ui-model-output --mode native-window

uv run --no-sync flashdreams-run-v2 slangpy-ui-invoke-async --mode webrtc
```

Open the URL the WebRTC window prints. The live renderer needs CUDA,
Vulkan/CUDA interop, and SlangPy, which is why these are the only integrations
here that will not run on a CPU. Native-window mode presents the result directly
in a GLFW window and keeps GPU-resident tensors off the CPU.

#### Writing one

Subclass `SlangPyUILoop` from `flashdreams.runtime_v2.slangpy_ui_loop` and
implement `step_ui(ui, step_index, events)` rather than `step`. The `ui` argument
exposes `ui.screen`, the root `slangpy.ui.Screen` that receives top-level
widgets, and every public type from `slangpy.ui` alongside it.

FlashDreams delegates those names straight through rather than wrapping them, so
the [SlangPy UI API reference](https://slangpy.shader-slang.org/en/stable/src/api_reference.html#ui)
is the source of truth for every widget constructor, method, property, flag and
callback.
