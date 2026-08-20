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
  and so cannot be asked for a particular picture.

An integration is then a factory:

```python
class SelfForcingT2VApplication(T2VApplication):
    def __init__(self, pipeline_config: Any | None = None) -> None:
        defaults = T2VApplicationDefaults.from_runner_config(RUNNER_WAN21_T2V_1PT3B)
        ...
```

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
