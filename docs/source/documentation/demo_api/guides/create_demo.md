---
title: 'Create a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A demo is a reusable, model-independent `flashdreams.api_v2` application under
`apps/<demo>/`. It owns interaction, session lifecycle, input handling, UI, and
presentation policy. It does not own checkpoints or import a model integration.

Before adding one, check the existing [demo gallery](../../../demos/index.md). If T2V,
Cam2V, Action2V, or V2V already matches the interaction, add a
[model adapter](integrate_model.md) instead.

## 1. Create the package

```text

apps/<demo>/
  <demo>/
    __init__.py
    application.py
    session.py
    ui.py                 # only when custom composition or controls are needed
  tests/
  pyproject.toml
  README.md               # one-line link to the canonical docs page

```

The package depends on `flashdreams`, never on `integrations_v2`. Add optional
`local-window` or `serving` extras only when the demo directly needs them. The
root workspace already includes `apps/*`.

## 2. Implement the api_v2 lifecycle

Implement these responsibilities in order:

1. `IApplication.init` parses arguments after `--` and initializes state shared
   by sessions.
2. `IApplication.session_desc` cheaply describes native output before `init`.
3. `IApplication.create_session` validates the requested description and
   returns an uninitialized `ISession`.
4. `ISession.init` creates per-run state and registers one `IModelLoop`.
5. `IModelLoop.step` returns `list[StepResult]`; `is_finished` ends finite runs,
   and `reset` clears run-owned state.
6. Register an `IUILoop` only when the default model-output blit is insufficient.

A minimal session registration looks like:

```python

class DemoSession(ISession):
    def init(self) -> None:
        self.register_model_loop(DemoModelLoop, state=DemoModelState(...))

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

```

Keep expensive shared objects on the application. Keep caches, counters, and
other rollout state on the session's model-loop state. The UI and model loops
run on different threads; use `invoke_async` instead of mutating the other
loop's state directly.

See the [Demo Application API reference](../api_reference/application.md) for method contracts and
`apps/t2v/` for the reference reusable application.

## 3. Define integration hooks

Expose a small immutable defaults object or constructor arguments for the facts
a model adapter supplies, such as:

- pipeline config or model factory;
- native width, height, frame rate, and tensor layout;
- rollout length and default device;
- model-specific input resolver callbacks.

Do not hard-code a concrete integration or checkpoint. The adapter layer will
supply these values.

## 4. Test with a stand-in model

Tests live in `apps/<demo>/tests/` and must not import `integrations_v2`.
Exercise the app with a fake pipeline or state object and cover:

- argument parsing and validation;
- session-description validation;
- loop registration, step shape, reset, finish, and close behavior;
- scripted input for interactive demos;
- the full runtime path when a CPU stand-in makes that practical.

Mark the module `pytest.mark.ci_cpu`. Put real checkpoints and model-specific
behavior in the integration's tests instead.

## 5. Document the demo

Add `docs/source/demos/<demo>.md` with its purpose, controls, runtime
modes, and placeholder launch shape. Add it to `demos/.nav.yml` and link the
package README to that canonical page. A demo becomes publicly launchable only
when a model adapter registers an application slug.

After the model binding is complete, follow
[Offline Program Packager](../../tools/offline_program_packager.md) to build a
distributable folder that does not require Python, `uv`, or network access.

## Verify

```bash

uv sync --package flashdreams-<demo> --extra dev --inexact
uv run --no-sync pytest apps/<demo>/tests -m ci_cpu

```
