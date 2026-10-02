# MODEL

[Options index](../OPTIONS.md)

The selected runner chooses a model preset. The fields below are editable overrides of that preset: **SAVE** writes changed values to `config.yaml`, and the application uses them when it next starts. A mismatched checkpoint, network shape, latent format, or scheduler can prevent startup or change generated video. The pipeline's internal name is fixed by the runner and is not editable.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Device:** | `model.device` | Crazy Robotaxi | `--device` | Device used for the world model, normally `cuda`. |

## PIPELINE

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Synthetic Text Max Length:** | `model.pipeline.synthetic_text_max_length` | OmniDreams | — | Token count for synthetic latency measurements; blank for real runs. |

### DIFFUSION MODEL

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Seed:** | `model.pipeline.diffusion_model.seed` | FlashDreams | `--model-seed`, `--seed` | Seed for initial model noise and scheduler sampling; blank uses the global RNG. |
| **Context Noise:** | `model.pipeline.diffusion_model.context_noise` | FlashDreams | — | Timestep used when updating the autoregressive cache; `0` skips added noise. |
| **Noise In Unpatchified Shape:** | `model.pipeline.diffusion_model.noise_in_unpatchified_shape` | FlashDreams | — | Debug option that draws noise before patchifying to match another implementation's random sequence. |

## Component guides

| MODEL section | Guide |
| --- | --- |
| TRANSFORMER and NETWORK | [Transformer](model-transformer.md) |
| SELF ATTN OPTIMIZED IMPL CONFIG and CROSS ATTN OPTIMIZED IMPL CONFIG | [Optimized attention](model-attention.md) |
| SCHEDULER | [Scheduler](model-scheduler.md) |
| TEXT ENCODER, IMAGE ENCODER, ENCODER, and DECODER | [Encoders and decoder](model-encoders.md) |
