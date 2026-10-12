---
title: 'Application slugs and model adapters'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

An application slug identifies one installed, zero-argument
`flashdreams.api_v2.IApplication` factory. It selects both an interaction and a
model binding; it is not a pipeline-config registry key.

## Register an application

Register factories in the model integration's `pyproject.toml`:

```toml
[project.entry-points."flashdreams.applications_v2"]
"t2v-self-forcing-wan2.1-t2v-1.3b" = "self_forcing.apps.t2v.adapter:create_app"
"t2v-self-forcing-wan2.1-t2v-1.3b-taehv" = "self_forcing.apps.t2v.adapter:create_app_taehv"
```

Each target must be callable without arguments and return an uninitialized
`IApplication`. Importing the module or calling the factory must not load a
checkpoint, initialize CUDA, or open a window; `IApplication.init` owns that
work.

Use a stable, lowercase, hyphenated slug. Existing integrations generally use
`<demo>-<model-or-config>` and append a suffix for a supported variant. The
entry-point table is the source of truth; slug parsing is not an API.

## Discover and inspect slugs

```bash
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 \
    t2v-self-forcing-wan2.1-t2v-1.3b -- --help
```

The first command lists registered entry points. The second constructs the
application and asks its argument parser for help. Arguments before `--` belong
to `flashdreams-run-v2`; arguments after it belong to the application.

For small development-only examples, the registry can also import a module
whose name is the slug with hyphens changed to underscores and call its
`create_app`. This fallback is not listed by `--help`, so real integrations
should register an entry point.

## Keep ownership one-way

| Concern | Owner |
| --- | --- |
| Interaction, session, controls, UI, presentation policy | `apps/<demo>/` |
| Model implementation, checkpoint mapping, pipeline configs | `integrations_v2/<model>/` |
| Binding and application factory | `integrations_v2/<model>/apps/<demo>/adapter.py` |
| File, browser, or native-window transport | `flashdreams.runtime_v2` |

Do not copy a reusable app into the model package or add `runner.py`,
`launch.py`, or `model_session.py` merely to expose a v2 slug. Follow
[Integrate a model with a demo](integrate_model.md) for the adapter workflow.
