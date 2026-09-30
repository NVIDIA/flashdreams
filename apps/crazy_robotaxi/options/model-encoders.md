# MODEL — ENCODERS AND DECODER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

The following sections appear under **PIPELINE** on **MODEL**.

## TEXT ENCODER

Encodes prompts for the world model.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Model Name:** | `model.pipeline.text_encoder.model_name` | FlashDreams | — | Hugging Face model ID for Cosmos-Reason1. |
| **Revision:** | `model.pipeline.text_encoder.revision` | FlashDreams | — | Pinned model revision for Cosmos-Reason1. |
| **Max Length:** | `model.pipeline.text_encoder.max_length` | FlashDreams | — | Text token limit, including padding and truncation. |
| **Dtype:** | `model.pipeline.text_encoder.dtype` | FlashDreams | — | Parameter and activation precision of the text encoder. |
| **Embedding Concat Strategy:** | `model.pipeline.text_encoder.embedding_concat_strategy` | FlashDreams | — | How text layers are combined into embeddings. |
| **N Layers Per Group:** | `model.pipeline.text_encoder.n_layers_per_group` | FlashDreams | — | Group size for grouped text-layer pooling. |
| **Run On Cpu:** | `model.pipeline.text_encoder.run_on_cpu` | FlashDreams | — | Keeps the text encoder on the host CPU to save GPU memory. |
| **Embedding Cache Size:** | `model.pipeline.text_encoder.embedding_cache_size` | FlashDreams | — | Number of previously encoded prompt batches retained; `0` disables caching. |

## IMAGE ENCODER

The image encoder handles the first frame; the encoder handles per-chunk HD-map input. The standard runner uses Wan VAE components.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Checkpoint Path:** | `model.pipeline.image_encoder.checkpoint_path` | FlashDreams | — | Checkpoint for the Wan VAE encoder. |
| **Dtype:** | `model.pipeline.image_encoder.dtype` | FlashDreams | — | Parameter and activation precision of the Wan VAE encoder. |
| **Use Cuda Graph:** | `model.pipeline.image_encoder.use_cuda_graph` | FlashDreams | — | Enables CUDA graph replay for the VAE encoder. |
| **Use Compile:** | `model.pipeline.image_encoder.use_compile` | FlashDreams | — | Enables `torch.compile` for the VAE encoder. |
| **Base Dim:** | `model.pipeline.image_encoder.base_dim` | FlashDreams | — | Base VAE channel count; must match the checkpoint. |
| **Z Dim:** | `model.pipeline.image_encoder.z_dim` | FlashDreams | — | Latent channel count; must match the checkpoint. |
| **Patch Size:** | `model.pipeline.image_encoder.patch_size` | FlashDreams | — | Outer spatial patch factor; must match the checkpoint. |
| **Is Residual:** | `model.pipeline.image_encoder.is_residual` | FlashDreams | — | Selects the residual-block VAE architecture. |
| **Latent Mean:** | `model.pipeline.image_encoder.latent_mean` | FlashDreams | — | Per-channel latent normalization mean. |
| **Latent Std:** | `model.pipeline.image_encoder.latent_std` | FlashDreams | — | Per-channel latent normalization standard deviation. |

The OmniDreams image encoder also exposes native VAE controls. The selected preset determines whether acceleration is enabled:

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Native Vae Acceleration:** | `model.pipeline.image_encoder.native_vae_acceleration` | OmniDreams | — | Native VAE policy: `disabled`, `auto`, or `required`. |
| **Native Vae Build Root:** | `model.pipeline.image_encoder.native_vae_build_root` | OmniDreams | — | Native extension build and cache directory. |
| **Native Vae Max Jobs:** | `model.pipeline.image_encoder.native_vae_max_jobs` | OmniDreams | — | Parallel job cap for the native build. |
| **Native Vae Verbose Build:** | `model.pipeline.image_encoder.native_vae_verbose_build` | OmniDreams | — | Enables detailed native build logs. |
| **Native Vae Backend:** | `model.pipeline.image_encoder.native_vae_backend` | OmniDreams | — | Native VAE compute backend, currently `fp8`. |
| **Native Vae Fp8 State Path:** | `model.pipeline.image_encoder.native_vae_fp8_state_path` | OmniDreams | — | File containing the exported FP8 VAE state. |
| **Native Vae Fp8 Auto Export:** | `model.pipeline.image_encoder.native_vae_fp8_auto_export` | OmniDreams | — | Automatically exports FP8 state when needed. |

## ENCODER

The image encoder handles the first frame; the encoder handles per-chunk HD-map input. The standard runner uses Wan VAE components.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Checkpoint Path:** | `model.pipeline.encoder.checkpoint_path` | FlashDreams | — | Checkpoint for the Wan VAE encoder. |
| **Dtype:** | `model.pipeline.encoder.dtype` | FlashDreams | — | Parameter and activation precision of the Wan VAE encoder. |
| **Use Cuda Graph:** | `model.pipeline.encoder.use_cuda_graph` | FlashDreams | — | Enables CUDA graph replay for the VAE encoder. |
| **Use Compile:** | `model.pipeline.encoder.use_compile` | FlashDreams | — | Enables `torch.compile` for the VAE encoder. |
| **Base Dim:** | `model.pipeline.encoder.base_dim` | FlashDreams | — | Base VAE channel count; must match the checkpoint. |
| **Z Dim:** | `model.pipeline.encoder.z_dim` | FlashDreams | — | Latent channel count; must match the checkpoint. |
| **Patch Size:** | `model.pipeline.encoder.patch_size` | FlashDreams | — | Outer spatial patch factor; must match the checkpoint. |
| **Is Residual:** | `model.pipeline.encoder.is_residual` | FlashDreams | — | Selects the residual-block VAE architecture. |
| **Latent Mean:** | `model.pipeline.encoder.latent_mean` | FlashDreams | — | Per-channel latent normalization mean. |
| **Latent Std:** | `model.pipeline.encoder.latent_std` | FlashDreams | — | Per-channel latent normalization standard deviation. |

The OmniDreams encoder also exposes native VAE controls. The selected preset determines whether acceleration is enabled:

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Native Vae Acceleration:** | `model.pipeline.encoder.native_vae_acceleration` | OmniDreams | — | Native VAE policy: `disabled`, `auto`, or `required`. |
| **Native Vae Build Root:** | `model.pipeline.encoder.native_vae_build_root` | OmniDreams | — | Native extension build and cache directory. |
| **Native Vae Max Jobs:** | `model.pipeline.encoder.native_vae_max_jobs` | OmniDreams | — | Parallel job cap for the native build. |
| **Native Vae Verbose Build:** | `model.pipeline.encoder.native_vae_verbose_build` | OmniDreams | — | Enables detailed native build logs. |
| **Native Vae Backend:** | `model.pipeline.encoder.native_vae_backend` | OmniDreams | — | Native VAE compute backend, currently `fp8`. |
| **Native Vae Fp8 State Path:** | `model.pipeline.encoder.native_vae_fp8_state_path` | OmniDreams | — | File containing the exported FP8 VAE state. |
| **Native Vae Fp8 Auto Export:** | `model.pipeline.encoder.native_vae_fp8_auto_export` | OmniDreams | — | Automatically exports FP8 state when needed. |

## DECODER

Turns generated latents into video. The Crazy Robotaxi OmniDreams runners use a TAEHV decoder.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Checkpoint Path:** | `model.pipeline.decoder.checkpoint_path` | FlashDreams | — | Checkpoint for the video decoder. |
| **Dtype:** | `model.pipeline.decoder.dtype` | FlashDreams | — | Parameter and activation precision of the video decoder. |
| **Use Cuda Graph:** | `model.pipeline.decoder.use_cuda_graph` | FlashDreams | — | Enables CUDA graph replay for the video decoder. |
| **Use Compile:** | `model.pipeline.decoder.use_compile` | FlashDreams | — | Enables `torch.compile` for the video decoder. |
