---
title: 'Application slugs and model adapters'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This guide covers discovery for applications implementing
`flashdreams.api_v2` and launched by `flashdreams-run-v2`. See the
[Demo Application API reference](../api_reference/application.md) for the
application contracts.

## Discovery

Each integration registers zero-argument application factories in
`flashdreams.applications_v2`:

```toml

[project.entry-points."flashdreams.applications_v2"]
"cam2v-lingbot" = "lingbot.apps.cam2v.adapter:create_app"
"cam2v-lingbot-world-fast" = "lingbot.apps.cam2v.adapter:create_app_fast"

```

List the slugs installed in the current environment, then inspect one
application's arguments:

```bash

uv run flashdreams-run-v2 --help
uv run flashdreams-run-v2 cam2v-lingbot -- --help

```

The default slug is normally `<application>-<model>`. Compatible variants
append a descriptive suffix and map to an explicit factory in the same adapter.
The registered entry points in each integration's `pyproject.toml` are the
source of truth.

## Ownership

Reusable application behavior belongs under `apps/<application>/`. Model
implementation, configuration, and tests belong under
`integrations_v2/<model>/`. The only bridge is the small
`apps/<application>/adapter.py` module in the model package.

Do not add `runner.py`, `launch.py`, `runtime.py`, `model_session.py`,
or a model-specific copy of an existing application merely to make a v2 slug.
See [Integrate a model with a demo](integrate_model.md) for the workflow.
