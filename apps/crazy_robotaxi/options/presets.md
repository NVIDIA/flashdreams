# Crazy Robotaxi runner presets

[Options index](../OPTIONS.md) · [MODEL guide](model.md) · [App README](../README.md)

A runner preset supplies the starting model configuration and render resolution.
The Options menu exposes those values as editable settings. This guide explains
what each preset selects; the component guides explain each individual field.

## Presets at a glance

Every suffix below follows `crazy-robotaxi-omnidreams`. For example, `-fast-perf`
means `crazy-robotaxi-omnidreams-fast-perf`. “Standard” means the runner without a
suffix. “Based on” identifies the preset whose values are inherited before the
changes described below are applied.

| Suffix | Based on | Render size (pixels) | What it selects |
| --- | --- | --- | --- |
| (none) | Standard | 1280 × 704 | OmniDreams attention, BF16 input encoders, and the `[1000, 500]` denoising schedule. |
| `-perf` | Standard | 1168 × 640 | Required native FP8 DiT with cuDNN attention, the `[1000, 100]` schedule, and no separate cache-finalization pass. |
| `-fast-perf` | `-perf` | 1168 × 640 | Required native FP8 input encoders and an unset model seed. |
| `-rtx-5090` | `-perf` | 1168 × 640 | CPU text encoding, a smaller history window, and a preference for SageAttention-3 FP8. Intended to fit a 32 GB RTX 5090. |
| `-rtx-5090-fast` | `-fast-perf` | 1024 × 560 | The same RTX 5090 memory and attention choices, plus a smaller render size and native FP8 input encoders. |
| `-optimized-gb300` | Standard | 1280 × 704 | Optimized self- and cross-attention policies for GB300, with the `[1000, 100]` schedule. |
| `-optimized-rtx-pro-6000` | Standard | 1280 × 704 | Optimized self-attention with FP8 projections for RTX PRO 6000, with the `[1000, 100]` schedule. |
| `-responsive` | Standard | 1280 × 704 | Short visual history in the first nine transformer blocks and cache-relative RoPE. |
| `-perf-responsive` | `-perf` | 1168 × 640 | Responsive history with the performance schedule; native DiT is disabled. |
| `-fast-perf-responsive` | `-fast-perf` | 1168 × 640 | Responsive history and native FP8 input encoders; native DiT is disabled. |
| `-optimized-gb300-responsive` | `-optimized-gb300` | 1280 × 704 | Responsive history with the GB300 optimized attention policy. |
| `-optimized-rtx-pro-6000-responsive` | `-optimized-rtx-pro-6000` | 1280 × 704 | Responsive history with the RTX PRO 6000 optimized attention policy. |

These names describe configuration choices. Achieved frame rate and memory use
also depend on the device, native build, resolution, and saved overrides.

## Standard and shared settings

The standard runner uses the single-view distilled OmniDreams checkpoint,
Cosmos-Reason1 text encoding, LightVAE for both the first-frame image encoder and
the per-chunk HD-map encoder, and LightTAE through the TAEHV video decoder.
Every preset inherits these model assets.

The following values provide a reference for the preset differences:

| Options location / label | `config.yaml` key | Standard value |
| --- | --- | --- |
| MODEL → DIFFUSION MODEL → **Seed:** | `model.pipeline.diffusion_model.seed` | `42` |
| MODEL → TRANSFORMER → **Window Size T:** | `model.pipeline.diffusion_model.transformer.window_size_t` | `6` |
| MODEL → TRANSFORMER → **Early Short History Block Count:** | `model.pipeline.diffusion_model.transformer.early_short_history_block_count` | `null` |
| MODEL → TRANSFORMER → **Skip Finalize Kv Cache:** | `model.pipeline.diffusion_model.transformer.skip_finalize_kv_cache` | `false` |
| MODEL → TRANSFORMER → **Native Dit Acceleration:** | `model.pipeline.diffusion_model.transformer.native_dit_acceleration` | `disabled` |
| MODEL → NETWORK → **Apply Rope Before Kvcache:** | `model.pipeline.diffusion_model.transformer.network.apply_rope_before_kvcache` | `true` |
| MODEL → NETWORK → **Self Attention Backend:** | `model.pipeline.diffusion_model.transformer.network.self_attention_backend` | `omnidreams` |
| MODEL → NETWORK → **Cross Attention Backend:** | `model.pipeline.diffusion_model.transformer.network.cross_attention_backend` | `omnidreams` |
| MODEL → SCHEDULER → **Denoising Timesteps:** | `model.pipeline.diffusion_model.scheduler.denoising_timesteps` | `[1000, 500]` |
| MODEL → TEXT ENCODER → **Run On Cpu:** | `model.pipeline.text_encoder.run_on_cpu` | `false` |
| MODEL → TEXT ENCODER → **Embedding Cache Size:** | `model.pipeline.text_encoder.embedding_cache_size` | `0` |

All twelve presets use two inference steps, `len_t=2`, `sink_size_t=0`,
`guidance_scale=1.0`, and `context_noise=128`. They keep the transformer's
**Dtype:** at `bfloat16`, **Compile Network:** at `false`, and **Use Cuda Graph:**
at `true`. Native and optimized FP8 policies select quantized operations
separately from the transformer's top-level dtype.

The standard input encoders use `bfloat16`, `use_compile=false`,
`use_cuda_graph=true`, and `native_vae_acceleration=disabled`. The video decoder
uses `bfloat16`, `use_compile=false`, and `use_cuda_graph=true` in every preset.
See [Encoders and decoder](model-encoders.md), [Transformer](model-transformer.md),
and [Scheduler](model-scheduler.md) for these fields.

### Render resolution and display resolution

The render sizes in the overview set **RENDERER → RASTER → Width:** and
**Height:** (`renderer.raster.width` and `renderer.raster.height`). They determine
the semantic camera input and model output resolution. Application flags
`--width` and `--height` override them.

Every preset leaves **PRESENTATION → Width:** and **Height:** blank
(`presentation.width=null`, `presentation.height=null`), so the display follows
the model size. `--display-width` and `--display-height` can change the display
size independently. Increasing the display size does not increase the model's
generation resolution. See [Renderer](renderer.md) and [Presentation](presentation.md).

## `-perf`: native DiT and the performance schedule

This preset derives from standard and lowers the render size to 1168 × 640.
Under **MODEL**, it selects the following values:

| Options section / label | `config.yaml` key | Value |
| --- | --- | --- |
| TRANSFORMER → **Skip Finalize Kv Cache:** | `model.pipeline.diffusion_model.transformer.skip_finalize_kv_cache` | `true` |
| TRANSFORMER → **Native Dit Acceleration:** | `model.pipeline.diffusion_model.transformer.native_dit_acceleration` | `required` |
| TRANSFORMER → **Native Dit Backend:** | `model.pipeline.diffusion_model.transformer.native_dit_backend` | `fp8_kvcache_cudnn` |
| TRANSFORMER → **Native Dit Attention Backend:** | `model.pipeline.diffusion_model.transformer.native_dit_attention_backend` | `cudnn` |
| SCHEDULER → **Denoising Timesteps:** | `model.pipeline.diffusion_model.scheduler.denoising_timesteps` | `[1000, 100]` |

The schedule still has two steps; the second timestep changes from `500` to
`100`. Skipping cache finalization removes the separate post-denoising cache
update. Both choices can change generated video.

`required` means startup fails if the native DiT path is unavailable or its
configuration is unsupported. **NETWORK → Self Attention Backend:** and
**Cross Attention Backend:** remain `omnidreams` in the settings tree, but the
native DiT path uses **Native Dit Attention Backend:**. The optimized attention
controls described below apply to the non-native network path.

## `-fast-perf`: native input encoding

This preset inherits `-perf`, including its native DiT, scheduler, history,
decoder, and 1168 × 640 render size. It sets
`model.pipeline.diffusion_model.seed=null` (**DIFFUSION MODEL → Seed:**), which
uses the global random-number generator instead of a preset model seed.
`--model-seed` can supply an explicit seed.

It also applies the same values to both **IMAGE ENCODER** and **ENCODER**:

| Options label | Key under `model.pipeline.image_encoder` and `model.pipeline.encoder` | Value |
| --- | --- | --- |
| **Dtype:** | `dtype` | `float16` |
| **Use Compile:** | `use_compile` | `false` |
| **Use Cuda Graph:** | `use_cuda_graph` | `false` |
| **Native Vae Acceleration:** | `native_vae_acceleration` | `required` |
| **Native Vae Backend:** | `native_vae_backend` | `fp8` |
| **Native Vae Fp8 State Path:** | `native_vae_fp8_state_path` | `null` |
| **Native Vae Fp8 Auto Export:** | `native_vae_fp8_auto_export` | `true` |

For example, the image encoder dtype key is `model.pipeline.image_encoder.dtype`.
The `float16` setting describes the wrapper's dtype; `native_vae_backend=fp8`
selects native FP8 computation. The first-frame and HD-map inputs use this
acceleration. The **DECODER** remains the shared BF16 TAEHV decoder.

Automatic export prepares and caches the FP8 LightVAE state when needed. With a
blank state path, `OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH` can select the file;
otherwise the default is `artifacts/native_vae/lightvae_fp8_state.pt`. Native VAE
acceleration is required, so an unavailable encoder backend fails startup.

## `-rtx-5090` and `-rtx-5090-fast`: memory and attention choices

`-rtx-5090` inherits `-perf` at 1168 × 640. `-rtx-5090-fast` inherits
`-fast-perf` and lowers the render size to 1024 × 560. Both apply these values:

| Options section / label | `config.yaml` key | Value |
| --- | --- | --- |
| TEXT ENCODER → **Run On Cpu:** | `model.pipeline.text_encoder.run_on_cpu` | `true` |
| TEXT ENCODER → **Embedding Cache Size:** | `model.pipeline.text_encoder.embedding_cache_size` | `8` |
| TRANSFORMER → **Window Size T:** | `model.pipeline.diffusion_model.transformer.window_size_t` | `4` |
| TRANSFORMER → **Native Dit Attention Backend:** | `model.pipeline.diffusion_model.transformer.native_dit_attention_backend` | `prefer_sage3_fp8` |

CPU text encoding frees GPU memory used by Cosmos-Reason1. The cache retains
eight prompt batches to avoid repeatedly encoding the same text. A smaller
history window reduces attention-cache memory and changes the visual history
available to the model.

`prefer_sage3_fp8` selects SageAttention-3 FP8 when the native build and device
support it, and falls back to cuDNN otherwise, including Windows builds. This
attention fallback does not relax the inherited `native_dit_acceleration=required`.

The `-fast` variant keeps the native FP8 input encoders and unset model seed
from `-fast-perf`; the other RTX 5090 preset keeps BF16 input encoders and seed
`42`. Both keep the `[1000, 100]` schedule and shared video decoder.

## `-optimized-gb300` and `-optimized-rtx-pro-6000`: network attention policies

These presets derive from standard at 1280 × 704. Both set
`model.pipeline.diffusion_model.transformer.skip_finalize_kv_cache=true` and
`model.pipeline.diffusion_model.scheduler.denoising_timesteps=[1000, 100]`.
Native DiT remains disabled, so the **NETWORK** attention fields select the
execution policy. BF16 input encoders and seed `42` are inherited.

The following keys are under `model.pipeline.diffusion_model.transformer.network`.
The nested implementation fields appear in **SELF ATTN OPTIMIZED IMPL CONFIG**
or **CROSS ATTN OPTIMIZED IMPL CONFIG**; projection and SDPA quantization fields
appear under **QUANTIZATION**. See [Optimized attention](model-attention.md).

| Options label / section | Key under NETWORK | `-optimized-gb300` | `-optimized-rtx-pro-6000` |
| --- | --- | --- | --- |
| **Self Attention Backend:** | `self_attention_backend` | `optimized` | `optimized` |
| **Cross Attention Backend:** | `cross_attention_backend` | `optimized` | `omnidreams` |
| Self → **Qkv Fusion Option:** | `self_attn_optimized_impl_config.qkv_fusion_option` | `full` | `full` |
| Self → **Sdpa Backend:** | `self_attn_optimized_impl_config.sdpa_backend` | `cudnn` | `fa2` |
| Self → **Use Tma:** | `self_attn_optimized_impl_config.use_tma` | `false` | `true` |
| Self → **Projection:** | `self_attn_optimized_impl_config.quantization.projection` | `null` | `float8_e4m3fn` |
| Self → **Quantized Sdpa:** | `self_attn_optimized_impl_config.quantization.quantized_sdpa` | `true` | `true` |
| Cross → **Qkv Fusion Option:** | `cross_attn_optimized_impl_config.qkv_fusion_option` | `fuse_kv` | `fuse_kv` |
| Cross → **Sdpa Backend:** | `cross_attn_optimized_impl_config.sdpa_backend` | `fa2` | `fa2` |
| Cross → **Use Tma:** | `cross_attn_optimized_impl_config.use_tma` | `true` | `true` |
| Cross → **Projection:** | `cross_attn_optimized_impl_config.quantization.projection` | `null` | `null` |
| Cross → **Quantized Sdpa:** | `cross_attn_optimized_impl_config.quantization.quantized_sdpa` | `false` | `false` |

For GB300, self-attention quantizes Q/K/V for cuDNN SDPA while keeping projection
precision unchanged; cross-attention uses FlashAttention-2 without quantized
SDPA. The RTX PRO 6000 policy uses FP8 self-attention projections and quantized
FlashAttention-2 SDPA. Its cross-attention uses the OmniDreams backend, so the
cross optimized implementation fields are present but inactive.

## The five `-responsive` variants: short early-block history

Each responsive preset inherits the matching preset without `-responsive`,
including its resolution, seed, schedule, input encoders, decoder, attention
policy, and cache-finalization choice. It applies these values:

| Options section / label | `config.yaml` key | Value |
| --- | --- | --- |
| TRANSFORMER → **Window Size T:** | `model.pipeline.diffusion_model.transformer.window_size_t` | `4` |
| TRANSFORMER → **Early Short History Block Count:** | `model.pipeline.diffusion_model.transformer.early_short_history_block_count` | `9` |
| TRANSFORMER → **Native Dit Acceleration:** | `model.pipeline.diffusion_model.transformer.native_dit_acceleration` | `disabled` |
| NETWORK → **Apply Rope Before Kvcache:** | `model.pipeline.diffusion_model.transformer.network.apply_rope_before_kvcache` | `false` |

The first nine of the model's 28 transformer blocks use a one-chunk visual
history window; the remaining blocks use the configured full window. **Window
Size T:** is measured in latent temporal units before patchification, with two
latent frames per chunk in these presets. The policy aims to reduce early-layer
dependence on older images. Cache-relative RoPE makes key positions relative to
the retained cache.

Native DiT does not support early short history or cache-relative RoPE, so it
must stay disabled for this combination. Consequently, `-perf-responsive` keeps
the performance schedule but executes the OmniDreams network attention path.
`-fast-perf-responsive` also keeps its native FP8 input encoders. The two
optimized responsive variants keep their respective optimized network policies.
The inherited native DiT backend fields in responsive performance presets are
inactive while native DiT is disabled.

There are no registered `-rtx-5090-responsive` or `-rtx-5090-fast-responsive`
presets. The RTX 5090 presets' smaller window alone does not enable the early
short-history policy or cache-relative RoPE.

## Saved settings and preset comparisons

Startup applies settings in this order:

1. The selected runner's defaults, including the pipeline and render size.
2. Sparse overrides from the user YAML, selected by `--config PATH` or the
   default Crazy Robotaxi settings path.
3. Explicit application CLI arguments after `--`.

Switching runners keeps the saved overrides. For example, a saved
`renderer.raster.width` and `height` override the RTX 5090 preset's smaller
resolution, and a saved `model.pipeline.diffusion_model.seed` overrides the
blank seed in `-fast-perf`. **RESET TO DEFAULTS**, followed by **SAVE**, clears
the draft back to the currently selected runner's defaults; it affects every
Options page. Model and render settings take effect after a process restart.

To compare unmodified presets, use a separate configuration file containing
only `schema_version: 1`, and omit application setting overrides. For example,
with that file saved as `/tmp/crazy-robotaxi-presets.yaml`:

```bash
uv run --package flashdreams-omnidreams flashdreams-run-v2 \
  crazy-robotaxi-omnidreams-fast-perf --mode native-window -- \
  --config /tmp/crazy-robotaxi-presets.yaml
```

The [Options index](../OPTIONS.md#editing-and-saving) explains saving and reset
behavior. [Launch arguments](launch.md) covers configuration-file selection;
the component guides list the fields with direct CLI overrides.

## Configuration sources

The [pipeline presets](../../../integrations_v2/omnidreams/config.py) define the
model policies. The [Crazy Robotaxi adapter](../../../integrations_v2/omnidreams/apps/crazy_robotaxi/adapter.py)
supplies application render sizes, and the [integration entry points](../../../integrations_v2/omnidreams/pyproject.toml)
register the twelve runner names. [User settings](../crazy_robotaxi/settings.py)
combine those defaults with YAML overrides.
