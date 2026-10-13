---
title: 'Integrate a model with a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A demo integration is a thin adapter between a model config under
`integrations_v2/<model>/` and a reusable application under `apps/<demo>/`.
The adapter supplies model-specific defaults and registers a zero-argument
`IApplication` factory. It does not duplicate either side.

## 1. Choose the existing interaction

Pick the app whose inputs and presentation match the model: T2V, Cam2V,
Action2V, V2V, Interactive Drive, or Crazy Robotaxi. Create a new app only when
none of them owns the required interaction or UI.

Add that app package to the integration's dependencies. The app remains
model-independent; only the adapter imports both the app and model config.

## 2. Add the adapter

```text
integrations_v2/<model>/apps/<demo>/
  __init__.py
  adapter.py
  README.md              # installation and launch details only
```

A T2V adapter can be this small:

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

Subclass the reusable application only when it exposes a hook the model really
needs, such as a different cache initializer or compile override. The factory
must not load weights or initialize CUDA.

## 3. Register a stable slug

```toml
[project.entry-points."flashdreams.applications_v2"]
"t2v-my-model" = "my_model.apps.t2v.adapter:create_app"
```

Use a readable `<demo>-<model-or-config>` slug. Register each supported variant
explicitly with its own factory. Do not add `flashdreams.runner_configs`, a
`runner.py`, or a model-specific copy of the app solely for v2 discovery.

## 4. Test the seam on CPU

Put adapter tests in `integrations_v2/<model>/tests/`, not under the reusable
app. Inject a stand-in config and verify:

- factory construction performs no model load;
- defaults report the correct size, rate, layout, and rollout length;
- initialize, generate, and finalize run in order;
- argument validation, reset, finish, and close work;
- every entry-point slug resolves to `IApplication`.

Use the app's shared testing helpers when available. For example, `t2v.testing`
provides a stand-in pipeline and end-to-end adapter checks.

## 5. Install and inspect

```bash
uv sync --package flashdreams-<model> --extra dev --inexact
uv run --no-sync pytest integrations_v2/<model>/tests -m ci_cpu
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 <demo>-<model> -- --help
```

A real launch uses the same slug with `--mode mp4`, `webrtc`, or
`native-window`. Runtime options precede `--`; application options follow it.

## 6. Document once

Put installation, supported bindings, and canonical commands on the model page.
Link to the demo page for shared controls. Keep the adapter README limited to
package-specific installation and launch details.

See [Application slugs](application_slugs.md) for discovery details and
[Demo configuration](configuration.md) for ownership of overrides.
