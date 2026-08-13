# FlashDreams App Host

`flashdreams-app` is the model-neutral application host. It finds a provider
distribution in the active Python environment, asks it for a declarative
pipeline application spec, constructs the runtime, creates a session, runs the
session loop, and writes presentation artifacts itself.

```bash
uv run flashdreams-app t2v-app mp4 --output o.mp4 --prompt "A waterfall"
uv run flashdreams-app t2v-app webrtc --prompt "A waterfall"
```

## Provider contract

A compatible package must expose an importable module with exactly two public
entry points:

- `parse_options(parser, argv)`, which extends the selected mode's host parser
  with provider-specific flags, parses the remaining arguments, and returns the
  parsed values as a mapping.
- `create_app_spec(request: flashdreams_app.AppRequest)`, returning a
  mode-aware `AppSpec` containing a shared `PipelineAppSpec` and either an
  `Mp4RunSpec` or `WebRTCRunSpec`.

The module structurally conforms to `flashdreams_app.AppProvider`; the host
validates this contract when it loads the installed provider.

`PipelineAppSpec` contains only mode-independent runtime data:

| Provider-supplied field | Purpose |
|---|---|
| `pipeline_config` | A `StreamInferencePipelineConfig`; the host calls `setup()` and owns the resulting pipeline. |
| `initialize_cache` | A callback receiving the constructed pipeline and session input; it returns the session cache. |

`AppSpec.config` is an `AppConfig` containing the model identity, video
dimensions, frame rate, and output tensor layout used by host-owned
presentation. `AppRequest` contains only the selected mode and parsed invocation
options passed into the provider.

Invocation data belongs to the selected presentation mode:

| Run spec | Required data |
|---|---|
| `Mp4RunSpec` | `initial_input` and the finite `total_steps` written to the file. |
| `WebRTCRunSpec` | `initial_input` for each live session; no finite step count. |

The following is a complete provider skeleton:

```python
import argparse
from collections.abc import Mapping, Sequence
from typing import Any

from flashdreams_app import (
    AppConfig,
    AppRequest,
    AppSpec,
    Mp4RunSpec,
    PipelineAppSpec,
    WebRTCRunSpec,
)
from flashdreams.runtime import InferenceInput
from my_model.config import MY_PIPELINE_CONFIG


def parse_options(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> Mapping[str, Any]:
    parser.add_argument("--prompt", required=True)
    return vars(parser.parse_args(argv))


def create_app_spec(request: AppRequest) -> AppSpec:
    prompt = str(request.options["prompt"])
    initial_input = InferenceInput(global_conditioning={"prompt": prompt})
    if request.mode == "mp4":
        run = Mp4RunSpec(initial_input=initial_input, total_steps=4)
    else:
        run = WebRTCRunSpec(initial_input=initial_input)
    return AppSpec(
        config=AppConfig(
            model_id="my-model",
            fps=24,
            output_layout="tchw",
            video_width=832,
            video_height=480,
        ),
        pipeline=PipelineAppSpec(
            pipeline_config=MY_PIPELINE_CONFIG,
            initialize_cache=_initialize_cache,
        ),
        run=run,
    )


def _initialize_cache(pipeline: Any, inputs: InferenceInput) -> object:
    prompt = str(inputs.global_conditioning["prompt"])
    return pipeline.initialize_cache(
        text=[prompt],
        image=None,
        height=60,
        width=104,
    )
```

The host first parses only the provider and presentation mode. It then gives the
provider a parser containing that mode's host-owned options and all remaining
arguments. The host owns process/distributed initialization, pipeline setup,
runtime/session lifecycle, `generate`/`finalize` stepping, MP4 writing, and
WebRTC serving. Providers only parse their options, select/configure a pipeline,
and describe how global conditioning initializes its cache. Pipeline-specific
execution options, including compilation and CUDA graphs, remain encapsulated
by the pipeline config and its `setup()` implementation.
