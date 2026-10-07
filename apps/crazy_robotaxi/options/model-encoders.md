# MODEL — ENCODERS AND DECODER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

The following sections appear under **PIPELINE** on **MODEL**.

The runner preset fixes text embedding layout, VAE architecture, latent normalization, and the native VAE compute backend. Checkpoint, precision, execution, cache, and build controls remain editable.

## TEXT ENCODER

Encodes prompts for the world model.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Model Name:** | FlashDreams | — | HF repo id of the underlying Qwen2.5-VL model. |
| **Revision:** | FlashDreams | — | HF commit hash to pin. |
| **Max Length:** | FlashDreams | — | Token length to pad/truncate to. |
| **Dtype:** | FlashDreams | — | Parameter and activation precision of the text encoder. |
| **Run On Cpu:** | FlashDreams | — | Keep the bf16 model on the host and run it there. |
| **Embedding Cache Size:** | FlashDreams | — | Number of most recently encoded prompt batches whose embeddings are kept. 0 disables caching. |

## IMAGE ENCODER

The image encoder handles the first frame; the encoder handles per-chunk HD-map input. The standard runner uses Wan VAE components.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Checkpoint Path:** | FlashDreams | — | Checkpoint for the Wan VAE encoder. |
| **Dtype:** | FlashDreams | — | Parameter and activation precision of the Wan VAE encoder. |
| **Use Cuda Graph:** | FlashDreams | — | Wrap the encoder forward in a CUDA graph for replay. |
| **Use Compile:** | FlashDreams | — | Compile the encoder with torch.compile; can increase peak VRAM usage. |

The OmniDreams image encoder also exposes native VAE controls. The selected preset determines whether acceleration is enabled:

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Native Vae Acceleration:** | OmniDreams | — | Native VAE policy: `disabled`, `auto`, or `required`. |
| **Native Vae Build Root:** | OmniDreams | — | Native extension build and cache directory. |
| **Native Vae Max Jobs:** | OmniDreams | — | Parallel job cap for the native build. |
| **Native Vae Verbose Build:** | OmniDreams | — | Enables detailed native build logs. |
| **Native Vae Fp8 State Path:** | OmniDreams | — | File containing the exported FP8 VAE state. |
| **Native Vae Fp8 Auto Export:** | OmniDreams | — | Automatically exports FP8 state when needed. |

## ENCODER

The image encoder handles the first frame; the encoder handles per-chunk HD-map input. The standard runner uses Wan VAE components.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Checkpoint Path:** | FlashDreams | — | Checkpoint for the Wan VAE encoder. |
| **Dtype:** | FlashDreams | — | Parameter and activation precision of the Wan VAE encoder. |
| **Use Cuda Graph:** | FlashDreams | — | Wrap the encoder forward in a CUDA graph for replay. |
| **Use Compile:** | FlashDreams | — | Compile the encoder with torch.compile; can increase peak VRAM usage. |

The OmniDreams encoder also exposes native VAE controls. The selected preset determines whether acceleration is enabled:

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Native Vae Acceleration:** | OmniDreams | — | Native VAE policy: `disabled`, `auto`, or `required`. |
| **Native Vae Build Root:** | OmniDreams | — | Native extension build and cache directory. |
| **Native Vae Max Jobs:** | OmniDreams | — | Parallel job cap for the native build. |
| **Native Vae Verbose Build:** | OmniDreams | — | Enables detailed native build logs. |
| **Native Vae Fp8 State Path:** | OmniDreams | — | File containing the exported FP8 VAE state. |
| **Native Vae Fp8 Auto Export:** | OmniDreams | — | Automatically exports FP8 state when needed. |

## DECODER

Turns generated latents into video. The Crazy Robotaxi OmniDreams runners use a TAEHV decoder.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Checkpoint Path:** | FlashDreams | — | Path to a pretrained TAEHV checkpoint. Defaults to the lighttae weights. |
| **Dtype:** | FlashDreams | — | Network parameter / activation dtype. |
| **Use Cuda Graph:** | FlashDreams | — | Wrap the decoder forward in a CUDA graph for replay. |
| **Use Compile:** | FlashDreams | — | torch.compile(mode="max-autotune-no-cudagraphs"). |
