---
title: 'Demo configuration'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams configuration uses strongly typed Python `dataclass` objects. A
demo selects a named, module-level pipeline definition from
`integrations_v2/<model>/config.py`, then derives a private copy for any
supported application overrides. This keeps model architecture separate from
UI, prompt, output-path, and serving policy.

This guide covers the complete Demo API configuration workflow. For the public
application lifecycle, see the [Demo Application API reference](../api_reference/application.md).

## 1. Start from the closest pipeline

Prefer an existing recipe or integration config. Use one explicit baseline
literal when the architecture differs materially, and use `derive_config` for
variants that change only a few fields:

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

```

`derive_config` deep-copies the supplied instance, applies nested dictionary
patches, and raises `KeyError` for an unknown field. It does not accept a config
class. Never mutate a shared baseline in place.

## 2. Understand the component configs

Configurable inference components (such as the encoder, transformer, scheduler,
and decoder) have corresponding configuration dataclasses. Their interfaces live
under `flashdreams.infra`; concrete reusable implementations live under
`flashdreams.recipes` or in an integration package.
As outlined in the
[stream inference pipeline guide](../../inferencing_api/guides/stream_inference_pipeline.md), the main
entry point for defining an integration is the
`~flashdreams.infra.pipeline.StreamInferencePipelineConfig`.

These config objects are modular and nestable.
A typical pipeline config defines the architecture by composing other config dataclasses:

```python

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.infra.pipeline import StreamInferencePipelineConfig

# Define your own configs for the encoder, transformer, and decoder
MyStreamingEncoderConfig = ...
MyTransformerConfig = ...
MyStreamingDecoderConfig = ...

# Compose them into a pipeline config
pipeline_config = StreamInferencePipelineConfig(
    name="customized-method-name",
    encoder=MyStreamingEncoderConfig(),
    diffusion_model=DiffusionModelConfig(
        transformer=MyTransformerConfig(),
        scheduler=FlowMatchSchedulerConfig(),
    ),
    decoder=MyStreamingDecoderConfig(),
)

```

## 3. Create a component config when needed

If you are interested in creating a brand new model component, you will need to create a corresponding config with the associated parameters you want to expose.

Let's say you want to create a new one-shot context encoder called `MyEncoder`.
You can create an `Encoder` subclass and a corresponding `MyEncoderConfig`
whose `_target` field points to that class. Pipeline `encoder` slots instead
require a `~flashdreams.infra.encoder.StreamingEncoder` and its
per-rollout cache contract.

```python

from dataclasses import dataclass, field
from flashdreams.infra.encoder.base import EncoderConfig, Encoder

@dataclass(kw_only=True)
class MyEncoderConfig(EncoderConfig):
    """My custom encoder config."""

    # Point to the class that will be instantiated by this config
    _target: type["MyEncoder"] = field(default_factory=lambda: MyEncoder)

    # Expose your configurable parameters
    embedding_dim: int = 512
    num_layers: int = 6

class MyEncoder(Encoder):
    """My custom encoder model.

    Args:
        config: Configuration to instantiate the encoder.
    """

    # Enable type checking
    config: MyEncoderConfig

    def __init__(self, config: MyEncoderConfig) -> None:
        super().__init__(config)

        # Build your layers using self.config.embedding_dim, etc.
        ...

    def forward(self, input):
        ...

```

## 4. Preserve model-defining fields

Review every inherited field, especially:

- checkpoint path and state-dict transform;
- temporal chunk length and native frame dimensions;
- scheduler and exact denoising timesteps;
- guidance and random seed;
- cache window, sink tokens, and image-latent stamping;
- precision, compilation, CUDA graphs, and attention backend.

A short config is useful only when its inherited behavior is intentional. Copy
fields explicitly when omission should fail loudly during review.

## 5. Export stable names

Give each public literal an uppercase name and a unique config `name`. When a
package provides several configs, export a lookup mapping:

```python

MY_MODEL_CONFIGS = {
    config.name: config
    for config in (PIPELINE_MY_MODEL, PIPELINE_MY_MODEL_FAST)
}

```

The demo adapter imports these literals directly. Do not add a global runner
registry or a second builder solely to avoid one config literal.

## 6. Apply demo overrides to a copy

A model config does not automatically become a public command line. A
`flashdreams.api_v2.IApplication` chooses the arguments it supports, validates
them in `init`, and derives a private config copy:

```python

pipeline_config = derive_config(
    MY_BASE_PIPELINE_CONFIG,
    diffusion_model={"seed": args.seed},
)

```

Only expose overrides that the application can validate and support. Runtime
arguments precede `--`; application arguments follow it. Inspect an
installed application's arguments with:

```bash

uv run flashdreams-run-v2 <demo>-<model> -- --help

```

## 7. Test before loading weights

Add `ci_cpu` tests that check:

- every exported config has the expected unique name and component types;
- derived variants leave the baseline unchanged;
- unknown override paths fail;
- checkpoint transforms produce the expected representative keys;
- `setup()` can be structurally inspected without downloading a checkpoint,
  when the component supports CPU or meta construction.

Use an explicitly opted-in GPU or manual test for real weights and a short
rollout. Continue with [Create a model](../../inferencing_api/guides/create_model.md)
for package and checkpoint responsibilities, or use the
[CLI Reference](../../cli.md) to launch
and inspect an installed application.
