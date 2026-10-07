# MODEL

[Options index](../OPTIONS.md)

The selected runner chooses a model preset, including its architecture and embedding formats. The fields below are editable overrides of that preset: **SAVE** writes changed values to `config.yaml`, and the application uses them when it next starts. Checkpoint overrides must match the preset's architecture. Performance and scheduler controls remain editable.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Device:** | Crazy Robotaxi | `--device` | Device used for the world model, normally `cuda`. |

## PIPELINE

### DIFFUSION MODEL

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Seed:** | FlashDreams | `--model-seed`, `--seed` | Seed for initial model noise and scheduler sampling; blank uses the global RNG. |
| **Context Noise:** | FlashDreams | — | Timestep used when updating the autoregressive cache; `0` skips added noise. |
| **Noise In Unpatchified Shape:** | FlashDreams | — | Debug option that draws noise before patchifying to match another implementation's random sequence. |

## Component guides

| MODEL section | Guide |
| --- | --- |
| TRANSFORMER and NETWORK | [Transformer](model-transformer.md) |
| SELF ATTN OPTIMIZED IMPL CONFIG and CROSS ATTN OPTIMIZED IMPL CONFIG | [Optimized attention](model-attention.md) |
| SCHEDULER | [Scheduler](model-scheduler.md) |
| TEXT ENCODER, IMAGE ENCODER, ENCODER, and DECODER | [Encoders and decoder](model-encoders.md) |
