---
title: 'Create a demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A demo implements `IApplication` and `ISession` from `flashdreams.api_v2`.
Reuse an app from the [demo gallery](../../../demos/index.md) when it already
supports your interaction; then [add a model adapter](integrate_model_adapter_for_a_demo.md).

This tutorial uses [Minimal T2V](../../../demos/minimal_t2v.md), an ImGui prompt
field connected to a model loop. Source snippets below come from the repository.

## Files

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
```

Keep model-specific pipeline construction and defaults in an integration adapter.
The app defines the pipeline contract and must work against a stub pipeline;
its tests must not import `integrations_v2`.

## Package

`apps/*` is already a workspace member. Define the app's dependencies and export
its application class. The model adapter registers the runnable entry point.

```toml
--8<-- "apps/minimal_t2v/pyproject.toml"
```

```python
--8<-- "apps/minimal_t2v/minimal_t2v/__init__.py"
```

## Application

[`IApplication`](../api_reference/application.md#flashdreams.api_v2.application.IApplication)
holds resources shared by its sessions. Keep the constructor and
`session_desc()` free of model loading: the runtime requests the description
before `init()`. Here `init()` parses arguments and creates the pipeline;
`create_session()` validates dimensions, layout, and frame rate before creating
an uninitialized session. `close()` releases the shared pipeline reference.

The adapter supplies `pipeline_config`, a `SessionDesc`, and `total_blocks`,
the maximum blocks **per prompt**. This app requires `tchw` output and a pipeline
with `initialize_cache`, `generate`, and `finalize` methods.

```python
--8<-- "apps/minimal_t2v/minimal_t2v/application.py"
```

## Session

[`ISession`](../api_reference/application.md#flashdreams.api_v2.session.ISession)
registers its model and UI loops during `init()`. Each loop owns its mutable
state. The model loop runs inference; the UI loop handles input and presentation.
Omit UI registration when the runtime's default frame display is sufficient.

```python
--8<-- "apps/minimal_t2v/minimal_t2v/session.py"
```

## Model loop

`TextToVideoState` keeps the prompt, cache, and per-prompt block index.
`TextToVideoLoop.step()` initializes the cache lazily, generates and finalizes
one block, and returns a `StepResult` with its frame count, layout, and metrics.
Empty prompts or completed rollouts return `[]` while waiting for another
prompt; the session remains open. `reset()` restarts the prompt, and `close()`
releases its cache.

```python
--8<-- "apps/minimal_t2v/minimal_t2v/model_loop.py"
```

## UI loop

`MinimalT2VUILoop.step_ui()` draws the prompt field and **Send** button.
`invoke_async` queues a prompt update on the model thread; capture the submitted
text so later edits cannot change that queued message. The UI presents the
latest model frame without directly modifying model state.

```python
--8<-- "apps/minimal_t2v/minimal_t2v/ui_loop.py"
```

## Check and document

Run the existing CPU checks against a stub pipeline, without downloading weights:

```bash
uv run --no-sync pytest apps/minimal_t2v/tests -m ci_cpu
```

After [adding an adapter](integrate_model_adapter_for_a_demo.md), put install
commands, requirements, and presets on its model page. Put controls and demo
arguments on `docs/source/demos/<demo>.md` and link to the model's quick start.
Add the demo to `demos/.nav.yml` and the matching category in `demos/index.md`.
For a new model, also update `models/.nav.yml`, `models/index.md`, and the home
gallery. Use [Interactive serving](interactive_serving.md#record-a-preview) to
record a preview, then [build the website](../../tools/build_website.md).
