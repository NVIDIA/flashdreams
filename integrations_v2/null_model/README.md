<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams NULL Model

**Not a v2 application.** This is a v1 pipeline — an encoder, transformer,
scheduler and decoder — whose output is arithmetic rather than video, so the
framework tests can drive a whole pipeline without a checkpoint. It implements
none of the v2 protocols and is not reachable through `flashdreams-run-v2`.

For a v2 application to copy, use
[`color_fade`](../color_fade/README.md) for the file path or
[`red_screen`](../red_screen/README.md) for the interactive one, and read the
[integration guide](../README.md). What follows is about the pipeline.

## Observable contract

| Property | Value |
| --- | --- |
| Input | Tensor with shape `[1, 1]` |
| Output shape | Tensor with shape `[1, 3, 1, 1, 1]` |
| Output value | `Input + cache.autoregressive_index` |
| Output layout | `VideoTensorLayout.bcthw` |

## Files

| File | |
| --- | --- |
| `config.py` | Defines the null-model pipeline. |
| `encoder.py` | Adds 100 to the input, as minor obfuscation. |
| `transformer.py` | Turns the encoded input into a flow for the scheduler to denoise. |
| `decoder.py` | Subtracts the 100 back off. |

```python
NULL_MODEL_CONFIG = NullModelConfig(
    name="null-model",
    encoder=NullInputEncoderConfig(),
    diffusion_model=DiffusionModelConfig(
        transformer=NullTransformerConfig(),
        scheduler=FlowMatchSchedulerConfig(
            num_inference_steps=1,
            denoising_timesteps=[1000],
        ),
    ),
    decoder=NullDecoderConfig(),
)
```

## How the pipeline is put together

### A real integration package

[`pyproject.toml`](pyproject.toml) declares `flashdreams-null-model` as a
workspace package depending on `flashdreams`. Keeping it isolated lets it declare
dependencies without affecting the other integrations.

### The per-step encoder

[`encoder.py`](null_model/encoder.py) defines `NullInputEncoder` as a
`StreamingEncoder`, bound in the config to `NullModelConfig.encoder`. A streaming
encoder runs on every autoregressive step, unlike
`NullModelConfig.diffusion_model.transformer.context_encoder`, which runs once at
the start of a generation.

It adds 100 to the 1x1 input tensor.

### The transformer

[`transformer.py`](null_model/transformer.py) defines, in order:

1. `latent_shape` — one batch, three channels, one frame, one pixel.
2. `initialize_autoregressive_cache()` — a cache tracking the autoregressive step
   of a continuous generation.
3. `initial_noise()` — zeros, so the scheduler has no noise to denoise beyond the
   flow `predict_flow` returns.
4. `predict_flow()` — the flow, from the encoded input and the step index the
   cache reports.

```text
# NullTransformer
target = encoded_input + cache.autoregressive_index
flow   = noisy_latent - target

# FlowMatchScheduler, sigma 1.0, one step
clean = noisy_latent - 1.0 * flow
      = noisy_latent - (noisy_latent - target)
      = target
```

Which is why one scheduler step is enough, and why every output tensor is exactly
the expected value.

### The per-step decoder

[`decoder.py`](null_model/decoder.py) defines `NullDecoder`, which subtracts the
encoder's 100 back off the transformer's output.

## Tests

```bash
uv run --no-sync pytest integrations_v2/null_model
```
