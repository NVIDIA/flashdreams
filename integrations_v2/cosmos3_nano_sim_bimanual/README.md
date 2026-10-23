# Cosmos3 Nano Sim Bimanual

The `cosmos3_nano_sim_bimanual` package is the complete
bimanual model integration for FlashDreams v2. It owns:

- checkpoint aliases, download/conversion, key remapping, and strict loading;
- the AgiBot MoT network, attention, RoPE, and bounded KV cache;
- action normalization/tokenization and first-frame encoding;
- the v2 action encoder, transformer, stateful video decoder,
  pipeline, and their configs.

The implementation uses FlashDreams' checkpoint readers, `BlockKVCache`, CUDA
graph wrapper, generic `DiffusionModel` and `FlowMatchScheduler`, Wan 2.2 VAE
encoder/decoder recipes, and TAEHV recipe. Only the checkpoint-specific MoT,
action/mRoPE layout, and KV-cache policy remain local.

The [HandDreams adapter](apps/handdreams/README.md) owns application
registration, model CLI/defaults, scene loading/encoding, and controller-action
encoding. It binds this model to the reusable
[HandDreams application](../../apps/handdreams/README.md), which owns
controller calibration, session orchestration, and UI. Dependencies point from
this integration to the app, never from the app to a model integration.

`PIPELINE_COSMOS3_NANO_SIM_BIMANUAL` uses the GA checkpoint and four-step sampler.
`HandTransformerConfig` owns checkpoint, cache, overlap, compile, CUDA graph,
profiling, and model-FPS controls; `HandDecoderConfig` owns decoder backend and
device settings. Diffusion seed and denoising schedule remain standard
`DiffusionModelConfig` and `FlowMatchSchedulerConfig` fields. App imports are confined to the
`apps/handdreams` adapter; the model implementation remains independent of it.

The HandDreams demo selects checkpoint alias `cosmos3_nano_sim_bimanual_ga`,
the iteration-1380 EMA export. It resolves to the transformer index in
[`nvidia/Cosmos3-Nano-Sim-Bimanual` at revision `304bf3dc45901793f0782254a5270cc087b9a157`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/304bf3dc45901793f0782254a5270cc087b9a157).
Do not substitute `main`: other revisions can use a different action contract.

FlashDreams' checkpoint loader downloads the six indexed shards (~30.3 GB),
using `HF_TOKEN` or an existing Hugging Face login, and caches a merged copy.
Allow space for both shard and merged caches (~60.7 GB total, plus other assets).
The integration remaps Diffusers parameter names without changing tensor values.
The demo uses the export's four-step SDE schedule: sigmas
`[1, 0.9375, 0.8333333333333334, 0.625]`, implemented by the existing shift-5
FlowMatch scheduler. Two latent frames produce eight video frames per step;
the attention window remains 64 latent frames.

GA actions use `unified_v1`: ego pose at `0:9`, right wrist at `9:18`, right
gripper width at `33`, left wrist at `34:43`, and left gripper width at `58`.
For action row `i`, the wrist target `P[i+1]` is relative to `P[16 * (i // 16)]`.
The anchor spans two eight-action steps and resets with the rollout, not with
the attention window. AgiBot gripper openness becomes a physical width
(`0.12 * openness` meters). Global-asinh normalization uses the exact exported
artifact SHA-256 `ae695e2c643a5626b3c3564f00705aa3a83ab0da7640549c7340db71172594b0`.
Missing fingertip slots are zeroed **after** normalization, then 59 values are
padded to 64. Robot controls remain domain 15; stationary ego rotation remains
valid. The training `camera_mask` augmentation is not an unconditional ego mask.
Scene FPS supplies the existing FPS modulation (the fetched Beta scenes use 30).

Old `agibot_*` aliases and recognized pre-GA exports are rejected: they require
the retired 29D framewise codec. Remove old `--checkpoint` and `--denoise-steps 2`
overrides. Explicit local or S3 exports must contain **GA weights**; matching
tensor shapes alone does not establish action-contract compatibility. The
default uses the revision-specific HF cache, not an old converted export.
S3 conversions use cache keys and stored provenance covering the full
source, EMA preference, and output dtype. Older conversion directories without
that provenance remain untouched and require an explicit local `--checkpoint`
path for reuse. Invalid conversion caches fail before downloading or overwriting
files; choose a new output directory or explicitly request `force=True` when
calling `fetch_and_convert` to rebuild one.
The demo's TAEHV2 Super decoder uses the author's public
[`taew2_2_super.safetensors`](https://github.com/madebyollin/taehv/blob/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taew2_2_super.safetensors),
pinned to commit `62f7591f59dfbb4c3c02b7a621d180a9eeaba26c`. The existing
FlashDreams atomic downloader validates decoder keys and shapes before caching
the file; later runs reuse it without a network request. Custom local, Hugging
Face, and S3 decoder checkpoints remain supported. The default HandDreams
startup needs no S3 credentials.

Scenario images and metadata are not packaged. By default the adapter prepares
three AgiBotWorld-Beta scenes in the user's cache using their Hugging Face login.
Explicit `--sample-dir` or `--scene-path` bypasses dataset fetching; see the
[scene preparation and format](apps/handdreams/README.md#scene-format).

## Validation

CPU tests cover the GA action contract and anchor transactions, checkpoint
remapping, scheduler math, transformer-cache behavior,
component ownership, lifecycle ordering, package isolation, and tensor shapes
without loading production checkpoints. The network and KV hot paths were
relocated without algorithmic changes. Legacy DCP exports retain the
`~/.cache/flashdreams/cosmos_interactive` checkpoint namespace; HF downloads use
the standard Hugging Face cache plus FlashDreams' merged-checkpoint cache.

GA conditioning uses the unchanged caption, a reflection-padded 832×480 VAE
canvas, and a 22×40 content latent (220 spatial tokens), producing 640×352 video.
CPU tests cover these conventions and controller/model FPS agreement. The
conditioning corrections still require a GA checkpoint quality test on hardware.

The following historical measurements used the **pre-GA** checkpoint and are
not GA quality or performance acceptance results. A direct-integration GB300
smoke run loaded the 15.17B-parameter two-step
checkpoint, encoded a then-bundled scene, and produced finite
`[8, 3, 360, 640]` pixel chunks through the Wan decoder. A separate uncapped,
model-only 40-step run used the optimized block/window-64, overlap, compile,
valid-length-attention, and CUDA-graph stack. After two hidden prewarm chunks,
steps 33–40 (past cache fill) had a median 109.95 ms per two-latent chunk, or
72.76 pixel-frame FPS. The same benchmark after this cleanup measured 110.44 ms
per chunk, or 72.43 FPS—within 0.6% of the 109.86 ms / 72.82 FPS pre-migration
baseline. These measurements exclude controller, decoder (for the model-only
run), UI, and transport pacing; they are not a legacy-output quality comparison.

GPU decoder smokes also loaded Wan, TAEHV, and TAEHV2 Super through the
registered `HandDecoder` backend. Each produced finite `[8, 3, 360, 640]`
chunks and repeated the same output from a fresh session cache without
reloading its weights.
