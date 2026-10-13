---
title: 'Create a model'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A model integration is a standalone workspace package under
`integrations_v2/<model>/`. It owns model code, checkpoint translation,
pipeline configs, and model-specific tests. It does not copy reusable demo,
UI, or transport behavior.

## 1. Scope before implementing

Record the facts that determine the integration:

- closest existing recipe or integration;
- checkpoint source, format, dtype, license, and access requirements;
- native spatial and temporal shape;
- scheduler, denoising steps, guidance, and random-seed behavior;
- streaming or one-shot execution, chunk length, window, and sink policy;
- model-specific conditioning, attention, memory, or decoder deltas;
- upstream revision, inputs, and command used for parity.

Reuse `flashdreams.core`, `flashdreams.infra`, and `flashdreams.recipes`. If the
model is a variant of an existing backbone, implement only the delta. Keep
model-specific branches out of shared framework layers.

## 2. Scaffold the package

```text
integrations_v2/<model>/
  __init__.py
  config.py                 # public pipeline literals
  impl/
    __init__.py
    checkpoint.py           # only when keys or envelopes need translation
    ...                     # model-specific implementation
  apps/                     # added when binding a reusable demo
    <demo>/adapter.py
  tests/
  pyproject.toml
  README.md
```

Use a distribution name such as `flashdreams-<model>` and depend on
`flashdreams`. Add a reusable app dependency only when adding its adapter. Do
not make `flashdreams` import the integration; dependency direction is
framework to recipe to integration.

## 3. Compose a stable pipeline

Start from the closest recipe and replace only the components that differ:

- a one-shot context encoder on the transformer;
- a per-step `StreamingEncoder` on the pipeline;
- transformer or network implementation;
- scheduler and exact denoising schedule;
- `StreamingDecoder`;
- checkpoint path and state-dict transform;
- a pipeline subclass only when cache initialization needs a genuinely
  different public signature.

Export one module-level config literal per supported variant and a name-keyed
mapping:

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

Use `derive_config` for small variants. Keep model-defining values explicit:
checkpoint, scheduler, steps, chunk and cache sizes, guidance, precision,
compile, CUDA graphs, and attention backend.

## 4. Prove checkpoint compatibility on CPU

A checkpoint transform should unwrap only known envelopes, remove known
prefixes, apply ordered key renames, and leave tensor values unchanged. Before
loading a GPU model, compare the transformed checkpoint with
`network.state_dict()` using CPU or meta tensors:

- no missing model keys;
- no unexpected checkpoint keys;
- identical shape for every matching key;
- representative real-key spot checks for every rename family.

Prefer checkpoint metadata or safetensors headers when full weights are too
large. If switching checkpoint sources, compare the transformed tensors from
both sources before claiming equivalence. A successful import is not enough;
unexplained missing or extra keys are integration failures.

## 5. Test the model boundary

Put model tests in `integrations_v2/<model>/tests/`. Every pytest test must have
exactly one of `ci_cpu`, `ci_gpu`, or `manual`.

Keep these checks CPU-safe when possible:

- package imports and metadata;
- config names, component types, and derived-variant independence;
- checkpoint key and shape bijection;
- small-tensor component shapes and cache/reset behavior;
- adapter-free pipeline flow with a stand-in;
- application adapter flow with the reusable app's testing helpers.

Use `ci_gpu` or `manual` for real checkpoints, generation, parity, quality, and
performance. Validate at least one step beyond initial cache fill for streaming
models, because rolling-window bugs often appear only after the boundary.

## 6. Bind and document after the model works

A model package can exist without a public application slug. Once its config
and model tests are stable, follow
[Integrate a model with a demo](../../demo_api/guides/integrate_model.md) to add
`integrations_v2/<model>/apps/<demo>/adapter.py` and a
`flashdreams.applications_v2` entry point.

Add or update `docs/source/models/<model>.md` with requirements, installation,
supported application slugs, canonical commands, and validated limitations.
Keep implementation and parity details in the package README or tests rather
than duplicating the public model page.

```bash
uv sync --package flashdreams-<model> --extra dev --inexact
uv run --no-sync pytest integrations_v2/<model>/tests -m ci_cpu
uv run --group lint pre-commit run -a
```

Default CPU verification must not download large checkpoints or start GPU
generation.
