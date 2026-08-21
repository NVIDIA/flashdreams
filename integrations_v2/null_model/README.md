<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams NULL Model
## Observable contract

| Property | Value |
| --- | --- |
| Input | Tensor with shape `[1, 1]` |
| Output shape | Tensor with shape `[1, 3, 1, 1, 1]` |
| Output value | Output is `Input + cache.autoregressive_index` |
| Output layout | `VideoTensorLayout.bcthw` |

## Files

`config.py`           Defines the 'null model' pipeline
`encoder.py`          Adds 100 to the input for minor obfuscation
`transformer.py`      Processes the encoded-input into a latent/flow that scheduler must 'denoise'
`decoder.py`          Removes the minor obfuscation created by the encoder by subtracting 100

```python
NULL_MODEL_CONFIG = NullModelConfig(
    name="null-model",
    encoder=NullInputEncoderConfig(), # Encoder of inputs
    diffusion_model=DiffusionModelConfig( # "Archtype" of our pipeline
        transformer=NullTransformerConfig(), # Transformer
        scheduler=FlowMatchSchedulerConfig( # Scheduler
            num_inference_steps=1,
            denoising_timesteps=[1000],
        ),
    ),
    decoder=NullDecoderConfig(), # Decoder of latents/output-tensor
)
```

## How the integration was designed, step by step

### Make a real integration package

[`pyproject.toml`](pyproject.toml) declares `flashdreams-null-model` as a
workspace package that depends on `flashdreams`. This issolated `pyproject.toml`
This keeps our integration isolated so it can safely declare dependencies without
affecting other integrations.

### Implement the per-step encoder

[`encoder.py`](null_model/encoder.py) defines `NullInputEncoder` as a
`StreamingEncoder` (bound in config to `NullModelConfig::encoder`).
This is important because it allows us to define an encoder that runs for each auto-regressive step.
This contrasts the `NullModelConfig::diffusion_model::transformer::context_encoder` which is designed to run only once at the beginning of a session for generation.

The encoder performs minor obfuscation by adding 100 to the 1 by 1 input tensor.

### Implement the transformer

In order the following were defined:
1. `latent_shape` declares one batch, three channels, one frame, and one pixel as the shape of the output tensor.
2. `initialize_autoregressive_cache()` creates a cache object which tracks the autoregressive step of each continuous generation.
3. `initial_noise()` returns zeros so that the scheduler does not have any 'noise' to 'denoise' from the `predict_flow` method result.
4. `predict_flow()` computes the flow utilizing our encoded_input tensor and the autoregressive step reported by the autoregressive cache.

The flow is computed as follows:
```text
# Explicit computation of our implemented `NullTransformer`
target = encoded_input + cache.autoregressive_index
flow   = noisy_latent - target

# `FlowMatchScheduler` implicit logic to denoise the flow into the final output tensor, sigma defaults to 1.0, 1 step.
clean = noisy_latent - 1.0 * flow
      = noisy_latent - (noisy_latent - target)
      = target
```

This is why one scheduler step is sufficient and why every output tensor is exactly the expected value.

### Implement the per-step decoder

[`decoder.py`](null_model/decoder.py) defines `NullDecoder`, which removes the
minor obfuscation by subtracting 100 from the transformer's output.

```bash
uv run --no-sync pytest integrations_v2/null_model
```
