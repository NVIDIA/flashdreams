---
title: 'Integrate model adapter for a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

An adapter is a package that contains the logic to connect a demo to a model. 

In this example we connect the [Minimal T2V application](create_demo.md) to Wan 2.1.

<a id="1-choose-the-existing-interaction"></a>

<a id="5-document-once"></a>

## Files to add or update

After creating `apps/minimal_t2v/`, add the adapter package and update the website:

```text
integrations_v2/wan21/
  config.py            # add a new pipeline config if desired
  apps/minimal_t2v/
    __init__.py       # add the package initializer
    adapter.py        # add the application factory
  pyproject.toml     # update dependencies, packages, and entry points

docs/source/
  models/
    wan21.md         # update: add Minimal T2V quick start, preset, and app-page link
    .nav.yml         # update when adding a new model page
    index.md         # add a card when adding a new model page
  demos/
    minimal_t2v.md   # update: link to the model's quick start
```

## Create the adapter entry point

Our adapter entry point is a factory function that creates an instance of our application based on the `IApplication` we defined.

This factory has the contract of:

```python
def some_create_app() -> IApplication:
    ...
```

In our case we then make in `integrations_v2/wan21/apps/minimal_t2v/adapter.py`:

```python
--8<-- "integrations_v2/wan21/apps/minimal_t2v/adapter.py"
```

## Register the application

In `integrations_v2/wan21/pyproject.toml`, add our application's package name to `tool.uv.sources`

Then add our `slug` to run our new application entry point to `[project.entry-points."flashdreams.applications_v2"]`

This is seen in `integrations_v2/wan21/pyproject.toml`:
```toml
--8<-- "integrations_v2/wan21/pyproject.toml"
```

## Install and run entry point

```bash
uv sync --package flashdreams-wan21 --inexact
uv run --no-sync flashdreams-run-v2 minimal-t2v-wan21 --mode native-window -- --default-prompt "A cat surfing"
```
## Adding a new pipeline configuration to pick from

Modify `integrations_v2/wan21/config.py` to add a new pipeline config to `wan21`:

```python
PIPELINE_WAN21_T2V_SEED123 = derive_config(
    PIPELINE_WAN21_T2V_1PT3B_480P,
    name="wan21-t2v-1.3b-480p-seed123",
    diffusion_model={"seed": 123},
)
```

This defines a new pipeline config that is a copy of `PIPELINE_WAN21_T2V_1PT3B_480P` with the addition of a `seed` parameter to the diffusion model, setting the pipeline to always run diffusion with the same seed of `123`.

To learn more about demo serving, see [Interactive serving](interactive_serving.md).