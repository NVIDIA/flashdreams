<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Text-to-video on the v2 API, for every t2v model rather than one of them.

Each text-to-video model takes a prompt and generates blocks of frames at a
size and rate it was trained for, so the command line for one is the command
line for all of them. This package owns that command line, and an integration
supplies only what differs.

- `defaults.py`: `T2VApplicationDefaults` is what an integration contributes,
  and `from_runner_config` reads it off the runner config the integration
  already has, so a model's frame size and rate are not written down twice.
- `application.py`: `T2VApplication` declares `--prompt`, `--pixel-width`,
  `--pixel-height`, `--fps`, `--total-blocks`, `--device`, and `--compile`,
  defaulting each from those defaults. `_configure_argument_parser` and
  `_apply_parsed_arguments` are where a model adds a flag of its own.
- `session.py`: `T2VSession` is one rollout, encoding the prompt into a cache
  and generating one autoregressive block per step.
- `testing.py`: test support, imported by an integration's tests and by nothing
  that runs in production. `check_t2v_model_impl` runs an application, measures
  the frames on their way to the file, and reports which expectations they
  missed — which is how a test says "that is a video" about a model that samples
  and so cannot be asked for a particular picture. `FakeT2VPipeline` is the
  stand-in those tests run on a CPU, and lives here because the contract it
  stands in for is `T2VSession`'s and so is the same for every model.

An integration is then a factory:

```python
class SelfForcingT2VApplication(T2VApplication):
    def __init__(self, pipeline_config: Any | None = None) -> None:
        defaults = T2VApplicationDefaults.from_runner_config(RUNNER_WAN21_T2V_1PT3B)
        ...
```

Five models are behind it, and between them they are why the parts above are
split the way they are:

| Application | Model | A run |
| --- | --- | --- |
| `t2v_self_forcing` | Self-Forcing Wan 2.1 1.3B, 480p | streams, 9 frames then 12 a block |
| `t2v_causal_forcing` | Causal-Forcing Wan 2.1 1.3B, 480p | streams, 9 frames then 12 a block |
| `t2v_fastvideo_causal_wan22` | CausalWan 2.2 14B, 480p | streams, two transformers |
| `t2v_wan21` | Wan 2.1 1.3B, 480p | one block, 81 frames |
| `t2v_cosmos_predict2` | Cosmos Predict2 2B, 720p | one block, 93 frames |

The last two are bidirectional: they attend over the whole clip and generate it
at once, so they override `_validate_total_blocks` to refuse a second block,
which would not continue the first. CausalWan 2.2 overrides
`_apply_compile_override`, since it denoises with two transformers and the shared
override reaches one. Nothing else differs, which is the point.

That is every model here that generates from a prompt alone. The rest condition
on something else — an image, a camera path, a pose, an HDMap — and `T2VSession`
forecloses all of them by passing `image=None` when it encodes the prompt. Wan
2.2 TI2V is the one to bring across next, being a prompt and a first frame and
nothing further: it needs a session that encodes that frame, and a way for the
frame to reach the session, which `create_session` does not currently offer. What
that hook should look like is worth settling on a model that needs it rather than
in advance.

`session_desc()` is the piece a runner needs and the protocol does not carry: a
caller has to describe a session before one exists to describe it, and only the
application knows what its model generates.
`flashdreams.runtime_v2.cli` calls it and passes the result straight back to
`create_session`, which is why `flashdreams-run-v2` needs no size flags of its
own. It is a method here rather than on `IApplication` because what a session
looks like before it exists is not settled for every kind of model, only for
this one.

`--total-blocks` says how long a rollout normally is, and bounds nothing by
itself: a v2 session cannot declare that it has finished, so the runner is told
how many steps to generate. The CLI uses this as that number when its caller
did not give one.
