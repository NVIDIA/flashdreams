---
title: 'Integrate a model with a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A demo integration is the thin adapter between a model config under
`integrations_v2/<model>/` and a reusable application under `apps/<demo>/`.
It supplies model-specific defaults and registers a zero-argument
`IApplication` factory.

## 1. Reuse the demo package

Choose the existing demo whose interaction matches the model: T2V, Cam2V,
Action2V, V2V, Interactive Drive, or Crazy Robotaxi. Create a new demo only when
none of these owns the required input, UI, or presentation behavior.

Add the demo package as a dependency of the integration. The app remains
model-independent; only the adapter imports both sides.

## 2. Add the adapter

```text

integrations_v2/<model>/apps/<demo>/
  __init__.py
  adapter.py
  README.md              # launch-only details

```

For the shared T2V application, the adapter shape is:

```python

from flashdreams.api_v2.application import IApplication
from t2v import T2VApplication, T2VApplicationDefaults

from my_model.config import PIPELINE_MY_MODEL

MY_MODEL_T2V_DEFAULTS = T2VApplicationDefaults(
    pipeline_config=PIPELINE_MY_MODEL,
    total_blocks=8,
    pixel_width=832,
    pixel_height=480,
    fps=16,
)


def create_app() -> IApplication:
    return T2VApplication(defaults=MY_MODEL_T2V_DEFAULTS)

```

Subclass the reusable application only when the model needs a real integration
hook that the app exposes. The factory takes no arguments, returns an
uninitialized application, and must not load checkpoints or initialize CUDA.

## 3. Register the factory

In the integration's `pyproject.toml`:

```toml

[project.entry-points."flashdreams.applications_v2"]
"t2v-my-model" = "my_model.apps.t2v.adapter:create_app"

```

Use `<demo>-<model>` for the default slug. Add a descriptive suffix and matching
`create_app_<suffix>` only for another supported config. Do not add
`flashdreams.runner_configs`, `runner.py`, or a model-specific copy of the demo.

## 4. Test the seam on CPU

Inject a stand-in pipeline config into the application and verify:

- the adapter declares native width, height, frame rate, layout, and rollout
  length correctly;
- the factory is importable and returns `IApplication` without loading a model;
- the demo drives initialize/generate/finalize in order;
- reset, finish, close, and argument validation work;
- the registered entry-point slug resolves.

Put these tests in `integrations_v2/<model>/tests/`, not under the demo package.
Use the shared demo's testing helpers when available; `t2v.testing` provides a
stand-in pipeline and end-to-end checks for T2V adapters.

## 5. Install and inspect

```bash

uv sync --package flashdreams-<model> --extra dev --inexact
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 <demo>-<model> -- --help

```

Runtime arguments precede `--`; demo arguments follow it. A real launch then
uses the same slug with `--mode mp4`, `webrtc`, or `native-window` as supported.

## 6. Document the binding

Add the command to the model page and link to the reusable demo page for shared
controls. Keep the adapter README limited to installation and launch details;
do not duplicate the demo's behavior or the model's implementation guide.

See [Application slugs](application_slugs.md) for discovery details.
