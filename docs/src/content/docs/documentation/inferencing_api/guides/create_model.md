---
title: 'Create a model'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

A model is a standalone package under `integrations_v2/<model>/`. It owns model
architecture, checkpoint mapping, pipeline behavior, stable configs, and
model-specific tests. It does not copy reusable demo behavior.

## 1. Choose the closest implementation

Before writing code, record:

- the nearest recipe or existing integration;
- checkpoint location, format, envelope, dtype, and access requirements;
- native spatial and temporal shape;
- scheduler, denoising steps, guidance, and cache policy;
- model-specific conditioning, attention, memory, or decoder changes;
- the upstream commit and inputs used for parity.

Reuse `flashdreams.core`, `flashdreams.infra`, and `flashdreams.recipes`.
Model-specific branches stay in the integration package rather than shared
framework code.

## 2. Scaffold the package

```text

integrations_v2/<model>/
  __init__.py
  config.py
  impl/
    __init__.py
    checkpoint.py        # when keys or envelopes need translation
    ...                  # model-specific implementation only
  tests/
  pyproject.toml
  README.md               # one-line link to the canonical model page

```

Use a distribution name such as `flashdreams-<model>` and depend on
`flashdreams`. Add only the nearest reusable app package when the adapter is
added later. Do not add a demo entry point while the model is still being
validated independently.

## 3. Implement only the model delta

Start from the closest recipe and replace only the encoder, transformer,
scheduler, decoder, checkpoint transform, or pipeline behavior that differs.
Keep public config literals in `config.py`; keep their implementation classes
under `impl/`.

Use the checkpoint helpers under `flashdreams.core.checkpoint`. A transform
should unwrap known envelopes, remove known prefixes, apply ordered renames,
and leave tensor values unchanged. Before loading on a GPU, compare transformed
checkpoint keys and shapes with `network.state_dict()` on CPU or meta tensors.
Missing, unexpected, and shape-mismatched keys should all be explained.

When exposing the model through a demo, follow
[Demo configuration](../../demo_api/guides/configuration.md) for the pipeline
definition.

## 4. Test the model boundary

Keep these checks CPU-safe when possible:

- package imports and metadata;
- config construction and variant independence;
- checkpoint key transforms using representative real key strings;
- component shape and cache/reset behavior with small tensors;
- adapter-free pipeline behavior against a stand-in.

Put real checkpoint loading, generation, parity, and performance checks behind
`ci_gpu` or `manual` as appropriate. Every pytest test requires a `ci_cpu`,
`ci_gpu`, or `manual` marker.

## 5. Bind it only after the model is stable

A model package can exist without a public application slug. When the config
and model tests are ready, follow
[Integrate a model with a demo](../../demo_api/guides/integrate_model.md)
to add `apps/<demo>/adapter.py`. The adapter owns presentation defaults and
registration; the model implementation does not import the demo.

## 6. Document and verify

Add or update `docs/src/content/docs/models/<model>.md` with requirements,
installation, supported demo bindings, and canonical commands. Keep detailed
implementation notes in the repository package page.

```bash

uv sync --package flashdreams-<model> --extra dev --inexact
uv run --no-sync pytest integrations_v2/<model>/tests -m ci_cpu
uv run --group lint pre-commit run -a

```

Do not download checkpoints or run GPU generation as part of the default CPU
workflow.
