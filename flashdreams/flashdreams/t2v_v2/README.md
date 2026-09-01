<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Text-to-video on the v2 API: one application every t2v model configures rather
than writes. A prompt can produce an MP4 or an interactive WebRTC session.

Every text-to-video model takes a prompt and generates blocks of frames at a
size and rate it was trained for, so the command line for one is the command
line for all of them. An integration supplies its defaults and inherits
everything else, which is why each of the five under `integrations_v2/` is one
subclass and a `pyproject.toml`.

## Run a model

```bash
export HF_TOKEN=<your-hf-token>
uv run --project integrations_v2/t2v_self_forcing flashdreams-run-v2 \
    t2v-self-forcing --output-path clip.mp4 \
    -- --prompt "A cat surfing" --total-blocks 7 --no-compile
```

The slug names the model, arguments before `--` describe the run, and arguments
after it go to the model. The first run fetches the checkpoint from Hugging Face,
tens of gigabytes including the text encoder, so set a token and expect to wait.
Writing an MP4 also needs `ffmpeg` on `PATH`.

To keep the model resident and submit prompts from the browser:

```bash
uv run --project integrations_v2/t2v_causal_forcing flashdreams-run-v2 \
    t2v-causal-forcing --mode webrtc --port 8080
```

Open `http://localhost:8080`. The application loads its model once, then shows
an ImGui prompt field. Each submission creates a fresh session and rollout
cache. Reaching `--total-blocks` leaves the final frame and prompt UI running;
it does not unload the model or stop the server. Refreshing the page reconnects
to the same process.

| Slug | Model | A run |
| --- | --- | --- |
| `t2v-self-forcing` | Self-Forcing Wan 2.1 1.3B, 480p | streams, 9 frames then 12 a block |
| `t2v-causal-forcing` | Causal-Forcing Wan 2.1 1.3B, 480p | streams, 9 frames then 12 a block |
| `t2v-fastvideo-causal-wan22` | CausalWan 2.2 14B, 480p | streams, two transformers |
| `t2v-wan21` | Wan 2.1 1.3B, 480p | one block, 81 frames |
| `t2v-cosmos-predict2` | Cosmos Predict2 2B, 720p | one block, 93 frames |

Each has a README of its own under `integrations_v2/`, with the frame arithmetic
for that model.

## Arguments

After the `--`, the same for every model, and listed by
`flashdreams-run-v2 SLUG -- --help`:

| Argument | |
| --- | --- |
| `--prompt` | Optional initial text to generate from. The UI supplies prompts for later sessions. |
| `--total-blocks` | Blocks to generate, which is how long the clip is. |
| `--device` | Device to load the model on. |
| `--compile` / `--no-compile` | Compile the network: minutes once, milliseconds a step. |
| `--seed` | Seed the noise, so the same command generates the same clip. |

Compilation is on in most of the model configs, so pass `--no-compile` for a
short clip rather than paying minutes to save milliseconds a block.

Before the `--` are `flashdreams-run-v2`'s own, including `--output-path`,
`--pixel-width`, `--pixel-height`, `--fps` and `--layout`. Unasked, a model
generates at the size and rate its checkpoint was trained for. Text-to-video
defaults to continuous presentation and every session uses its prompt UI.

## What a t2v run generates

`[-1, 1]` floats on whichever device `--device` loaded the model onto, in the
layout the model's runner config declares, `tchw` for all five today. The frame
size and rate are the checkpoint's, read off that runner config rather than
written down in the integration.

Another size can be asked for with `--pixel-width` and `--pixel-height`, as long
as each dimension is a multiple of 8, the decoder's spatial compression ratio. A
`--layout` the model does not emit is refused before a session starts.

`--fps` sets the rate the frames are meant to play at, and so the rate an MP4
plays back at. It is not a generation speed.

## What is here

- `defaults.py` - `T2VApplicationDefaults`, what an integration contributes.
  `from_runner_config` reads the frame size, rate, layout and rollout length off
  the runner config the model package already ships, so nothing is written twice.
- `application.py` - `T2VApplication`: the shared command line above, the model
  loaded once during application initialization and shared by every session,
  and the hooks an integration overrides.
- `session.py` - `T2VSession` and `T2VModelLoop`: one rollout. The session
  encodes the prompt into a per-run cache in `init` and registers the model loop;
  the loop generates one block per `step` and reports itself finished after
  `--total-blocks`. Interactive sessions also register the shared prompt UI; an
  initial session without a prompt has an already-finished model loop while that
  UI waits.
- `ui.py` - `T2VImGuiUILoop`: the prompt field and new-session button drawn over
  the latest generated frame.
- `testing.py` - the shared check an integration's tests call, and the stand-in
  pipeline they call it against on a CPU.

## Adding a model

Four things, of which only the first is more than a few lines.

**Subclass `T2VApplication`**, reading defaults off the runner config the model
package already ships:

```python
class Wan21T2VApplication(T2VApplication):
    def __init__(self, pipeline_config: Any | None = None) -> None:
        defaults = T2VApplicationDefaults.from_runner_config(
            RUNNER_WAN21_T2V_1PT3B_480P,
            total_blocks=1,
        )
        if pipeline_config is not None:
            defaults = dataclasses.replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)


def create_app() -> IApplication:
    return Wan21T2VApplication()
```

The optional `pipeline_config` is what lets the CPU tests substitute a stand-in.
`total_blocks` is passed explicitly only when the runner config states none,
which is the case for a model that does not roll out.

**Override a hook** if the model differs from the shared assumptions. All five
are on `T2VApplication`, and most integrations override none:

| Hook | Override it when |
| --- | --- |
| `_validate_total_blocks` | The model generates its whole clip at once, so a rollout has to be refused rather than quietly producing a second clip. |
| `_apply_compile_override` | `--compile` does not reach every network. CausalWan 2.2 has a high-noise and a low-noise transformer, and the shared override reaches one. |
| `_apply_seed_override` | The seed does not live on the diffusion model config. |
| `_configure_argument_parser` | The model takes an argument the shared five do not. |
| `_apply_parsed_arguments` | Something added by the hook above has to be kept. |

There is also `session_type`, unused today, for a model whose step is not one
autoregressive block and so needs a session other than `T2VSession`.

**Write the `pyproject.toml`**, depending on
`flashdreams[local-window,serving]` and the model package, and registering the
slug:

```toml
[project.entry-points."flashdreams.applications_v2"]
"t2v-wan21" = "t2v_wan21.app:create_app"
```

**Re-export `create_app`** from the package `__init__.py`, alongside the
application class the tests import.

## Testing a t2v integration

Two files, split by what they need.

`test_stand_in_model.py` is `ci_cpu` and covers only what is particular to this
integration, which model it runs, its defaults, and any hook it overrode. Pass
`FakeT2VPipelineConfig` in place of the checkpoint. For Wan 2.1 that is enough to
check the defaults came off the runner config and that a rollout is refused,
without generating anything.

To generate, hand the whole application to `check_t2v_model_impl`:

```python
pytestmark = pytest.mark.ci_cpu

def test_the_model_generates_from_a_prompt() -> None:
    pipeline = FakeT2VPipeline()
    result = check_t2v_model_impl(
        SelfForcingT2VApplication(pipeline_config=FakeT2VPipelineConfig(pipeline)),
        # The stand-in generates its own size rather than the checkpoint's, so
        # it says so here rather than asking the application.
        SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.ON_DEMAND,
            video_width=pipeline.width,
            video_height=pipeline.height,
        ),
        steps=3,
        commandline_args=["--prompt", _PROMPT, "--device", "cpu"],
        expected=ExpectedFrameStats(frame_count=33),
    )
    assert result.passed, result.failures
```

`check_t2v_model_impl` runs the whole path, the application initializes,
resolves a session and generates, and the frames are read the way an output sink
reads them, and returns what it measured alongside the expectations it missed.
`ExpectedFrameStats` checks a frame count, a mean luminance band and a minimum
change between frames, every field optional, since a model that samples can only
be expected to produce a picture, not a particular one.

Two details in that call are worth copying. `BLOCK` with `ON_DEMAND` is
what makes the frame count assertable, since every generated frame is then
presented exactly once. And `steps` is `run_session`'s own limit rather than
`--total-blocks`, so a run stops at whichever comes first, which is why a
stand-in test asking for more steps than the model's rollout length sees fewer
frames than it expected.

`test_real_model.py` is `ci_gpu` and downloads the checkpoint, so it asks to be
run rather than running automatically:

```python
pytestmark = pytest.mark.ci_gpu

_SKIP = real_model_run_skip_reason("T2V_WAN21_REAL_MODEL_RUN")

@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")
def test_the_model_generates_a_clip_worth_watching(tmp_path: Path) -> None:
    result = check_real_model_generates_a_clip(
        Wan21T2VApplication(),
        prompt=DEFAULT_PROMPT,
        steps=1,
        frame_count=81,
        mp4_path=tmp_path / "clip.mp4",
    )
    assert result.passed, result.failures
```

`real_model_run_skip_reason` returns why the run cannot happen here, no
environment variable, no GPU, no `ffmpeg`, or `None` when it can.

Each integration's own README carries the two commands that run these for its
package, writing the real-model clip under `$HOME` so that a sandboxed player can
open it afterwards. The
[integration guide](../../../integrations_v2/README.md#testing-it) covers
`--inexact` and the markers. General t2v behaviour is covered once in
`flashdreams/test_v2`, and a run reaching a real file once in the Self-Forcing
integration, so an integration's own tests repeat neither.

## Where to go next

- [Writing an integration](../../../integrations_v2/README.md) - for an
  application that is not text-to-video.
- [Architecture](../../../ARCHITECTURE.md) - how the layers a rollout runs
  through fit together.
- [v2 API protocols](../api_v2/README.md) - what `T2VApplication` implements.
- [Runtime](../runtime_v2/README.md) - what runs a rollout, and the command line
  that starts it.
- [`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json)
  and the [benchmark harness](../../tools/benchmarks/README.md) - comparing the
  models against each other.
