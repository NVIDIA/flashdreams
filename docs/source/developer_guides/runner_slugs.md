<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Application slugs and model adapter dispatch

Runnable applications live under `apps/`. Model integrations do not own demo
implementations or runner shims; they expose small bindings from
`integrations_v2/<model>/apps/<app>/adapter.py`.

## Application discovery

Each model package registers the applications it supports through the
`flashdreams.applications_v2` entry-point group. For example, OmniDreams binds
regular and performance-tuned variants of the reusable application:

```toml
[project.entry-points."flashdreams.applications_v2"]
"interactive-drive-omnidreams" = "omnidreams.apps.interactive_drive.adapter:create_app"
"interactive-drive-omnidreams-perf" = "omnidreams.apps.interactive_drive.adapter:create_perf_app"
```

Action2V can be bound by more than one model without duplicating the
application:

```toml
# integrations_v2/lingbot/pyproject.toml
"action2v-lingbot" = "lingbot.apps.action2v.adapter:create_app"

# integrations_v2/hy_worldplay/pyproject.toml
"action2v-hy-worldplay" = "hy_worldplay.apps.action2v.adapter:create_app"
```

## Ownership boundary

The reusable package owns UI behavior, argument parsing, application sessions,
and presentation. The model package owns its implementation and canonical
configuration:

```text
apps/<app>/
  pyproject.toml
  README.md
  <app>/
    ...
  tests/
    ...

integrations_v2/<model>/
  config.py
  impl/
    ...
  tests/
    ...
  apps/<app>/
    __init__.py
    adapter.py
    README.md
```

`config.py` is the model folder's unique pipeline config (or model-specific
pipeline-config wrapper) and the only implementation-facing module permitted at
the integration root. The adapter imports defaults from it, while all other
model implementation stays in `impl/`.

## Adding or changing a binding

1. Add reusable behavior to `apps/<app>/`.
2. Add model implementation to `integrations_v2/<model>/impl/`.
3. Define the model's pipeline configs, defaults, and hooks in
   `integrations_v2/<model>/config.py`.
4. Add the minimal adapter at
   `integrations_v2/<model>/apps/<app>/adapter.py`.
5. Register the default config under `flashdreams.applications_v2` as
   `<demo-slug>-<model-slug>` using `create_app`. Register additional variants
   as `<demo-slug>-<model-slug>-<suffix>` using `create_app_<suffix>`; for
   example, `t2v-foo-fast` maps to `create_app_fast`.
6. Run the CPU ownership and import checks:

```bash
uv run pytest -m ci_cpu \
  apps/t2v/tests/test_structure.py \
  integrations_v2/<model>/tests
```

Do not add `runner.py`, `launch.py`, `runtime.py`, `model_session.py`, or a
model-specific demo package to bridge the application and integration layers.
