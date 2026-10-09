---
title: 'Demo configuration'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams keeps model configuration separate from application and runtime
configuration. This separation lets one model serve several demos and lets one
demo run several models without either layer importing the other.

| Configuration | Owner | Examples |
| --- | --- | --- |
| Pipeline architecture | `integrations_v2/<model>/config.py` | checkpoint, scheduler, cache, encoder, decoder |
| Application defaults and arguments | app plus model adapter | prompt, block count, native size, device |
| Runtime and presentation | `flashdreams-run-v2` | output mode, host, port, output path, step limit |

## Define stable pipeline configs

Export one named, module-level config for each supported model variant. Derive
small variants from a baseline instead of mutating it:

```python
from typing import cast

from flashdreams.infra.config import derive_config
from flashdreams.recipes.wan import WanInferencePipelineConfig

PIPELINE_MY_MODEL = WanInferencePipelineConfig(
    name="my-model",
    encoder=...,
    diffusion_model=...,
    decoder=...,
)

PIPELINE_MY_MODEL_FAST = cast(
    WanInferencePipelineConfig,
    derive_config(
        PIPELINE_MY_MODEL,
        name="my-model-fast",
        diffusion_model={
            "scheduler": {"num_inference_steps": 4},
            "transformer": {"compile_network": True},
        },
    ),
)

MY_MODEL_CONFIGS = {
    config.name: config
    for config in (PIPELINE_MY_MODEL, PIPELINE_MY_MODEL_FAST)
}
```

`derive_config` deep-copies the baseline, applies nested patches, and raises
`KeyError` for an unknown field. Always give a derived public variant its own
`name`; never mutate a shared literal in place.

## Put components in the correct slots

A `StreamInferencePipelineConfig` composes three model-side stages:

- `encoder`: optional `StreamingEncoder` for per-step control, such as a camera
  trajectory, HD map, or input video chunk;
- `diffusion_model`: transformer plus scheduler;
- `decoder`: optional `StreamingDecoder` from clean latent to output.

A one-shot text or image context encoder belongs on the transformer's
`context_encoder`, not in the pipeline's per-step `encoder` slot. See the
[stream inference pipeline guide](../../inferencing_api/guides/stream_inference_pipeline.md)
for the lifecycle and cache boundaries.

## Apply application overrides to a copy

An application exposes only overrides it can validate. Parse them in
`IApplication.init`, then derive a private pipeline config before calling
`setup()`:

```python
pipeline_config = derive_config(
    PIPELINE_MY_MODEL,
    diffusion_model={"seed": args.seed},
)
pipeline = pipeline_config.setup().to(args.device).eval()
```

Keep output paths, browser settings, and presentation modes out of the model
config. They are runtime concerns. Keep prompts and UI policy out of shared
recipes. They are application concerns.

## Review and test the boundary

Before loading weights, verify on CPU that:

- public config names are unique and stable;
- derived variants leave the baseline unchanged;
- component types and model-defining fields are correct;
- unknown override paths fail;
- checkpoint key transforms handle representative real keys;
- each adapter factory returns an uninitialized application.

Use `ci_gpu` or `manual` checks for checkpoint downloads, real generation,
parity, and performance. Continue with [Create a model](../../inferencing_api/guides/create_model.md)
for model-package responsibilities or [Integrate a model with a demo](integrate_model.md)
for the application binding.
