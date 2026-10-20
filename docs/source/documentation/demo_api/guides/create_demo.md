---
title: 'Create a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A demo implements `IApplication` and `ISession` from `flashdreams.api_v2`.
Reuse an app from the [demo gallery](../../../demos/index.md) when it already
supports your interaction; then [add a model adapter](integrate_model_adapter_for_a_demo.md).

This tutorial builds the [Minimal T2V app](https://github.com/NVIDIA/flashdreams/tree/main/apps/minimal_t2v)
(`minimal_t2v`), which sends ImGui prompts to a model loop. All source below is included from the repository.

## New application files

```text
apps/minimal_t2v/
  pyproject.toml
  minimal_t2v/
    __init__.py
    application.py
    session.py
    model_loop.py
    ui_loop.py
  tests/test_application.py

docs/source/demos/minimal_t2v.md # Add the app page with controls, arguments, and a link to its model's quick start.
docs/source/demos/.nav.yml # Add `Minimal T2V: minimal_t2v.md` to the sidebar.
docs/source/demos/index.md # Add a Minimal T2V gallery card linking to `minimal_t2v/`.
```

## Package

Define a new demo package and export its application slug.

`apps/minimal_t2v/pyproject.toml`:
```toml
--8<-- "apps/minimal_t2v/pyproject.toml"
```

`apps/minimal_t2v/minimal_t2v/__init__.py`:
```python
--8<-- "apps/minimal_t2v/minimal_t2v/__init__.py"
```

## IApplication

Implement an [`IApplication`](../api_reference/application.md#flashdreams.api_v2.application.IApplication) to be implemented by *every model* that desires to integrate with this demo:

- `__init__` is your constructor/contract to any implementor of this demo.
    - `pipeline_config` is the model's pipeline configuration.
    - `desc` is our session description - more on this later.
    - `total_blocks` is the model's total number of blocks to generate for the entire session.
- `init()` parses arguments & is where we load/download our model.
- `session_desc()` is how our [`ISession`](../api_reference/application.md#flashdreams.api_v2.session.ISession) collects a session description from our IApplication.
    - [`SessionDesc`](../api_reference/application.md#flashdreams.runtime_v2.session_desc.SessionDesc)
- `create_session()` is called every time a session is made, passing `IApplication` data to `ISession`
- `close()` is called when the application is shutting down.
```python
--8<-- "apps/minimal_t2v/minimal_t2v/application.py"
```

## Session

[`ISession`](../api_reference/application.md#flashdreams.api_v2.session.ISession) owns the two main loops driving our application *session* (running in separate threads):

- [`IModelLoop`](../api_reference/application.md#flashdreams.api_v2.loop.IModelLoop): Drives model inference/logic.
- [`IUILoop`](../api_reference/application.md#flashdreams.api_v2.loop.IUILoop): Drives UI/user-interaction.

These are set with `register_*_loop()` methods part of `ISession`. These must be called during `ISession.init()`.

```python
--8<-- "apps/minimal_t2v/minimal_t2v/session.py"
```

## Model loop

`IModelLoop` governs all model logic and will be where we run our model inference:

- `MinimalT2VModelState` is the type of `ILoop.state` for our `IModelLoop`.
- `step()` is called for each loop of our `IModelLoop`.
    - `pipeline.initialize_cache()` is called to initialize our model's cache.
    - `pipeline.generate()` is called to generate a chunk of frames given our current model's state (`pipeline`).
    - `pipeline.finalize()` is called to finalize the generated chunk & produce metrics.
- `reset()` is called when the session is reset.
- `close()` is called when the session is closed.

```python
--8<-- "apps/minimal_t2v/minimal_t2v/model_loop.py"
```

## UI loop

`IUILoop` governs all UI logic and will be where we handle user-interaction:

- `MinimalT2VUIState` is the type of `ILoop.state` for our `IUILoop`.
- `invoke_async(self.state.model_loop, lambda...)` is called when the user submits a new prompt (via imgui button) to submit a `lambda` to run on our `IModelLoop` thread before the next new `IModelLoop.step()`.
- `step_ui(...)` is called for each loop of our `IUILoop`. It contains `imgui` state when `IUILoop` inherits from [`ImGuiUILoop`](../api_reference/application.md#flashdreams.runtime_v2.imgui_ui_loop.ImGuiUILoop).

```python
--8<-- "apps/minimal_t2v/minimal_t2v/ui_loop.py"
```
