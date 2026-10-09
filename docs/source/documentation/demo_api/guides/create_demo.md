---
title: 'Create a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A demo is a reusable, model-independent `flashdreams.api_v2` application under
`apps/<demo>/`. It owns interaction, session lifecycle, input handling, UI, and
presentation policy. It must not import a model integration or own a
checkpoint.

First check the [demo gallery](../../../demos/index.md). If T2V, Cam2V,
Action2V, V2V, Interactive Drive, or Crazy Robotaxi already owns the required
interaction, add a [model adapter](integrate_model.md) instead of another app.

## Package layout

```text
apps/<demo>/
  <demo>/
    __init__.py
    application.py
    defaults.py           # optional adapter-supplied defaults
    session.py
    ui.py                 # only for custom composition or controls
  tests/
  pyproject.toml
  README.md
```

Depend on `flashdreams`, never on `integrations_v2`. Add the `serving` or
`local-window` extra only when the app directly imports that optional stack.
The root workspace already discovers `apps/*` packages.

## Implement the lifecycle

The runtime calls the API in this order:

1. `IApplication.session_desc()` describes native output without loading the
   model.
2. `IApplication.init(args)` parses arguments after `--` and initializes state
   shared by sessions.
3. `IApplication.create_session(desc)` validates the requested description and
   returns an uninitialized `ISession`.
4. `ISession.init()` creates per-run state and registers exactly one
   `IModelLoop`; a custom `IUILoop` is optional.
5. `IModelLoop.step(index, events)` returns `list[StepResult]`.
6. `is_finished`, `reset`, and `close` manage the run lifecycle.

A minimal session registration is:

```python
from flashdreams.api_v2.session import ISession

class DemoSession(ISession):
    def init(self) -> None:
        self.register_model_loop(DemoModelLoop, state=DemoModelState(...))

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc
```

If the session does not register a UI loop, the runtime presents model output
with its default blit loop. Model and UI loops run on separate threads and own
their state; use `flashdreams.api_v2.loop.invoke_async` for cross-loop changes.

## Expose a narrow adapter seam

Put model-independent facts in the app and accept model-owned facts through a
small immutable defaults object or constructor arguments. Typical adapter
inputs are:

- pipeline config or model factory;
- native width, height, frame rate, and tensor layout;
- default rollout length and device;
- model-specific input resolver callbacks.

Keep expensive shared objects, such as the loaded pipeline, on the application.
Keep caches, counters, and resettable rollout state on the session's model-loop
state.

## Test without a real model

Tests live in `apps/<demo>/tests/` and must not import `integrations_v2`. Use a
stand-in pipeline to cover argument validation, session-description checks,
loop registration, step output, reset, finish, close, and scripted input. Mark
pure Python tests `ci_cpu`; put real-model behavior in the integration's tests.

```bash
uv sync --package flashdreams-<demo> --extra dev --no-default-groups --inexact
uv run --no-sync pytest apps/<demo>/tests -m ci_cpu
```

Add `docs/source/demos/<demo>.md`, update `demos/.nav.yml`, and keep the package
README focused on development. The demo becomes publicly runnable when a model
adapter registers an application slug.
