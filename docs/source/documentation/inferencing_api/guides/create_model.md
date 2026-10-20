---
title: 'Create a model'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A model integration is a workspace package in `integrations_v2/<model>/`.
It contains model code, checkpoint conversion, pipeline configs, and tests.
Reuse existing apps for demos, UI, and serving.

<a id="1-scope-before-implementing"></a>

## 1. Check the model requirements

Check the following before writing code:

- closest existing recipe or integration;
- checkpoint source, format, dtype, license, and access requirements;
- native spatial and temporal shape;
- scheduler, denoising steps, guidance, and random-seed behavior;
- streaming or one-shot execution, chunk length, window, and sink policy;
- differences in conditioning, attention, memory use, or decoding;
- upstream revision, inputs, and command used for parity.

Reuse `flashdreams.core`, `flashdreams.infra`, and `flashdreams.recipes`. If the
model is a variant of an existing backbone, implement only the differences. Keep
model-specific branches out of shared framework layers.

<a id="2-scaffold-the-package"></a>

## 2. Create the package

```text
integrations_v2/<model>/
  __init__.py
  config.py                 # pipeline configs
  impl/
    __init__.py
    checkpoint.py           # checkpoint conversion, if needed
    ...                     # model-specific implementation
  apps/                     # model adapters for existing apps
    <demo>/adapter.py
  tests/
  pyproject.toml

docs/source/models/<model>.md  # model documentation
```

Use a distribution name such as `flashdreams-<model>` and depend on
`flashdreams`. Add a reusable app dependency only when adding its adapter. The
integration can import `flashdreams`; `flashdreams` must not import the integration.

## 3. Configure the pipeline

Start from the closest recipe and replace only the components that differ:

- a one-shot context encoder on the transformer;
- a per-step `StreamingEncoder` on the pipeline;
- transformer or network implementation;
- scheduler and exact denoising schedule;
- `StreamingDecoder`;
- checkpoint path and state-dict transform;
- a pipeline subclass only when cache initialization needs different arguments.

Define a config for each supported variant and a dictionary keyed by config
name:

```python
from flashdreams.infra.pipeline import StreamInferencePipelineConfig

PIPELINE_MY_MODEL = StreamInferencePipelineConfig(
    name="my-model",
    encoder=...,
    diffusion_model=...,
    decoder=...,
)

MY_MODEL_CONFIGS = {
    config.name: config
    for config in (PIPELINE_MY_MODEL,)
}
```

Use `derive_config` for small variants. Set these values explicitly:
checkpoint, scheduler, steps, chunk and cache sizes, guidance, precision,
compile, CUDA graphs, and attention backend.

## 5. Test the integration

Put model tests in `integrations_v2/<model>/tests/`. Every pytest test must have
exactly one of `ci_cpu`, `ci_gpu`, or `manual`.

Keep these checks CPU-safe when possible:

- package imports and metadata;
- config names, component types, and derived-variant independence;
- matching checkpoint keys and tensor shapes;
- small-tensor component shapes and cache/reset behavior;
- pipeline execution with a fake model;
- application adapter flow with the reusable app's testing helpers.

Use `ci_gpu` or `manual` for real checkpoints, generation, parity, quality, and
performance. Validate at least one step beyond initial cache fill for streaming
models, because rolling-window bugs often appear only after the boundary.

Command to run all tests:
```bash
uv sync --package flashdreams-<model> --extra dev --inexact
uv run --no-sync pytest integrations_v2/<model>/tests -m ci_cpu
```

## 6. Add documentation

Add or update `docs/source/models/<model>.md` with the new model. Example: [Wan 2.1](../../../models/wan21.md)

## 7. [Optional] Add a demo adapter

A model package can exist without a demo implementation. To add one, follow the [integrate model adapter for a demo](../../demo_api/guides/integrate_model_adapter_for_a_demo.md) guide.
