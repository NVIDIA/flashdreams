# MODEL — ENCODERS AND DECODER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

The following sections appear under **PIPELINE** on **MODEL**.

The runner preset fixes text embedding layout, VAE architecture, latent normalization, and the native VAE compute backend. Checkpoint, precision, execution, cache, and build controls remain editable.

## TEXT ENCODER

Encodes prompts for the world model.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Model Name:** | FlashDreams | — | Hugging Face model ID for Cosmos-Reason1. |
| **Revision:** | FlashDreams | — | Pinned model revision for Cosmos-Reason1. |
| **Max Length:** | FlashDreams | — | Text token limit, including padding and truncation. |
| **Dtype:** | FlashDreams | — | Parameter and activation precision of the text encoder. |
| **Run On Cpu:** | FlashDreams | — | Keeps the text encoder on the host CPU to save GPU memory. |
| **Embedding Cache Size:** | FlashDreams | — | Number of previously encoded prompt batches retained; `0` disables caching. |

## IMAGE ENCODER

The image encoder handles the first frame; the encoder handles per-chunk HD-map input. The standard runner uses Wan VAE components.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Checkpoint Path:** | FlashDreams | — | Checkpoint for the Wan VAE encoder. |
| **Dtype:** | FlashDreams | — | Parameter and activation precision of the Wan VAE encoder. |
| **Use Cuda Graph:** | FlashDreams | — | Enables CUDA graph replay for the VAE encoder. |
| **Use Compile:** | FlashDreams | — | Enables `torch.compile` for the VAE encoder. |

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
| **Use Cuda Graph:** | FlashDreams | — | Enables CUDA graph replay for the VAE encoder. |
| **Use Compile:** | FlashDreams | — | Enables `torch.compile` for the VAE encoder. |

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
| **Checkpoint Path:** | FlashDreams | — | Checkpoint for the video decoder. |
| **Dtype:** | FlashDreams | — | Parameter and activation precision of the video decoder. |
| **Use Cuda Graph:** | FlashDreams | — | Enables CUDA graph replay for the video decoder. |
| **Use Compile:** | FlashDreams | — | Enables `torch.compile` for the video decoder. |
