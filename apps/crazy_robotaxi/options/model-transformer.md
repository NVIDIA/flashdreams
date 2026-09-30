# MODEL — TRANSFORMER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **TRANSFORMER** within **PIPELINE**, under **DIFFUSION MODEL**. **NETWORK** appears within **TRANSFORMER**.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Dtype:** | `model.pipeline.diffusion_model.transformer.dtype` | OmniDreams | — | Parameter and activation precision. |
| **Checkpoint Path:** | `model.pipeline.diffusion_model.transformer.checkpoint_path` | OmniDreams | — | Pretrained transformer weights. |
| **Batch Shape:** | `model.pipeline.diffusion_model.transformer.batch_shape` | OmniDreams | — | Batch dimensions of the generated latent. |
| **Num Views:** | `model.pipeline.diffusion_model.transformer.num_views` | OmniDreams | — | Number of camera views. |
| **Len T:** | `model.pipeline.diffusion_model.transformer.len_t` | OmniDreams | — | Latent frames generated per chunk. |
| **H Extrapolation Ratio:** | `model.pipeline.diffusion_model.transformer.h_extrapolation_ratio` | OmniDreams | — | Height RoPE extrapolation factor. |
| **W Extrapolation Ratio:** | `model.pipeline.diffusion_model.transformer.w_extrapolation_ratio` | OmniDreams | — | Width RoPE extrapolation factor. |
| **Window Size T:** | `model.pipeline.diffusion_model.transformer.window_size_t` | OmniDreams | — | Sliding self-attention history in temporal units before patchification. |
| **Sink Size T:** | `model.pipeline.diffusion_model.transformer.sink_size_t` | OmniDreams | — | Temporal sink tokens retained in attention history. |
| **Early Short History Block Count:** | `model.pipeline.diffusion_model.transformer.early_short_history_block_count` | OmniDreams | — | Number of first blocks limited to one chunk of visual history; blank disables this policy. |
| **Compile Network:** | `model.pipeline.diffusion_model.transformer.compile_network` | OmniDreams | `--compile`, `--no-compile` | Uses `torch.compile` for the network. |
| **Use Cuda Graph:** | `model.pipeline.diffusion_model.transformer.use_cuda_graph` | OmniDreams | — | Captures and replays steady-state model work in a CUDA graph. |
| **Cuda Graph Warmup Iters:** | `model.pipeline.diffusion_model.transformer.cuda_graph_warmup_iters` | OmniDreams | — | Eager warmup calls before graph capture. |
| **Skip Finalize Kv Cache:** | `model.pipeline.diffusion_model.transformer.skip_finalize_kv_cache` | OmniDreams | — | Skips the separate cache-finalization pass. |
| **Native Dit Acceleration:** | `model.pipeline.diffusion_model.transformer.native_dit_acceleration` | OmniDreams | — | Native DiT policy: `disabled`, `auto`, or `required`. |
| **Native Dit Build Root:** | `model.pipeline.diffusion_model.transformer.native_dit_build_root` | OmniDreams | — | Native extension build and cache location. |
| **Native Dit Max Jobs:** | `model.pipeline.diffusion_model.transformer.native_dit_max_jobs` | OmniDreams | — | Parallel job cap for the native extension build. |
| **Native Dit Verbose Build:** | `model.pipeline.diffusion_model.transformer.native_dit_verbose_build` | OmniDreams | — | Enables detailed native extension build logs. |
| **Native Dit Backend:** | `model.pipeline.diffusion_model.transformer.native_dit_backend` | OmniDreams | — | Native compute backend, `fp8_kvcache_cudnn` or `bf16`. |
| **Native Dit Attention Backend:** | `model.pipeline.diffusion_model.transformer.native_dit_attention_backend` | OmniDreams | — | Native attention implementation (`auto`, cuDNN/Sage/Sparge policies supported by the runtime). |
| **Native Dit Sparge Topk:** | `model.pipeline.diffusion_model.transformer.native_dit_sparge_topk` | OmniDreams | — | Top-k selection ratio for Sparge sparse attention. |
| **Native Dit Sparge Hybrid Period:** | `model.pipeline.diffusion_model.transformer.native_dit_sparge_hybrid_period` | OmniDreams | — | Period of a Sparge and SageAttention hybrid schedule. |
| **Native Dit Sparge Hybrid Phase:** | `model.pipeline.diffusion_model.transformer.native_dit_sparge_hybrid_phase` | OmniDreams | — | Phase offset for the hybrid sparse-attention schedule. |
| **Guidance Scale:** | `model.pipeline.diffusion_model.transformer.guidance_scale` | OmniDreams | — | Classifier-free guidance strength; `1` disables it. Higher values require negative text embeddings. |

## NETWORK

These fields define the model architecture and attention implementation.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **In Channels:** | `model.pipeline.diffusion_model.transformer.network.in_channels` | OmniDreams | — | Number of input latent channels. |
| **Out Channels:** | `model.pipeline.diffusion_model.transformer.network.out_channels` | OmniDreams | — | Number of output latent channels. |
| **Patch Spatial:** | `model.pipeline.diffusion_model.transformer.network.patch_spatial` | OmniDreams | — | Spatial patch size for height and width. |
| **Patch Temporal:** | `model.pipeline.diffusion_model.transformer.network.patch_temporal` | OmniDreams | — | Temporal patch size. |
| **Model Channels:** | `model.pipeline.diffusion_model.transformer.network.model_channels` | OmniDreams | — | Transformer hidden width. |
| **Num Blocks:** | `model.pipeline.diffusion_model.transformer.network.num_blocks` | OmniDreams | — | Number of transformer blocks. |
| **Num Heads:** | `model.pipeline.diffusion_model.transformer.network.num_heads` | OmniDreams | — | Number of attention heads. |
| **Mlp Ratio:** | `model.pipeline.diffusion_model.transformer.network.mlp_ratio` | OmniDreams | — | Feed-forward width relative to hidden width. |
| **Concat Padding Mask:** | `model.pipeline.diffusion_model.transformer.network.concat_padding_mask` | OmniDreams | — | Adds a padding-mask input channel. |
| **Use Adaln Lora:** | `model.pipeline.diffusion_model.transformer.network.use_adaln_lora` | OmniDreams | — | Enables low-rank adaptive layer normalization. |
| **Adaln Lora Dim:** | `model.pipeline.diffusion_model.transformer.network.adaln_lora_dim` | OmniDreams | — | Rank of the adaptive layer normalization LoRA. |
| **Use Crossattn Projection:** | `model.pipeline.diffusion_model.transformer.network.use_crossattn_projection` | OmniDreams | — | Projects text embeddings before cross-attention. |
| **Crossattn Proj In Channels:** | `model.pipeline.diffusion_model.transformer.network.crossattn_proj_in_channels` | OmniDreams | — | Input width of the text embedding projection. |
| **Crossattn Emb Channels:** | `model.pipeline.diffusion_model.transformer.network.crossattn_emb_channels` | OmniDreams | — | Key and value width for cross-attention. |
| **Timestep Scale:** | `model.pipeline.diffusion_model.transformer.network.timestep_scale` | OmniDreams | — | Multiplier applied before timestep embedding. |
| **Apply Rope Before Kvcache:** | `model.pipeline.diffusion_model.transformer.network.apply_rope_before_kvcache` | OmniDreams | — | Rotates keys before caching; off uses cache-relative RoPE. |
| **Additional Concat Ch:** | `model.pipeline.diffusion_model.transformer.network.additional_concat_ch` | OmniDreams | — | Extra channels for HD-map conditioning; zero disables that input. |
| **Enable Cross View Attn:** | `model.pipeline.diffusion_model.transformer.network.enable_cross_view_attn` | OmniDreams | — | Enables multi-view attention and view modulation. |
| **Cp Method:** | `model.pipeline.diffusion_model.transformer.network.cp_method` | OmniDreams | — | Context-parallel attention method: `ring` or `ulysses`. |
| **Self Attention Backend:** | `model.pipeline.diffusion_model.transformer.network.self_attention_backend` | OmniDreams | — | Implementation used by transformer self-attention. |
| **Cross Attention Backend:** | `model.pipeline.diffusion_model.transformer.network.cross_attention_backend` | OmniDreams | — | Implementation used by text and cross-view attention. |
| **View Condition Dim:** | `model.pipeline.diffusion_model.transformer.network.view_condition_dim` | OmniDreams | — | Width of the view-conditioning vector. |
| **N Cameras Emb:** | `model.pipeline.diffusion_model.transformer.network.n_cameras_emb` | OmniDreams | — | Number of camera-view embeddings. |

The optimized self-attention and cross-attention fields are listed in [Optimized attention](model-attention.md).
