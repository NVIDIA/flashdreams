# flashdreams-artifixer

ArtiFixer reconstruction-enhanced T2V inference for Wan 2.1 1.3B, packaged as
a [`flashdreams`](../..) plugin.

ArtiFixer extends Wan 2.1 1.3B with:

- per-block opacity and Plucker-camera-ray MLPs;
- a third KV bank for neighbor cross-attention with PRoPE;
- opacity-weighted latent mixing of noise with the VAE-encoded
  reconstruction-rendered frames;
- 4-step DMD distillation (`FlowMatchScheduler(shift=5)`).

The reference implementation lives in the
[dreamfix repo](https://gitlab-master.nvidia.com/hturki/dreamfix) under
`model_training/net/transformer.py` and `model_training/pipeline/`. This
plugin ports it to FlashDreams' faster Wan stack (RingAttention, cuDNN,
`torch.compile`, CUDA graphs).

## Components

| Component | Description |
| --- | --- |
| Recipe scaffold | AR rollout + 4-step DMD scheduler knobs match the dreamfix stage-3 run config. |
| Per-block conditioning | Opacity + camera-ray MLPs, neighbor cross-attention, and PRoPE — parity-tested against dreamfix. |
| Pipeline | Opacity-weighted latent mixing + self-forcing renoise loop inside `ArtifixerInferencePipeline.generate`. |
| dreamfix-compatible surface | `initialize_cache` accepts pre-encoded UMT5 prompts + VAE-encoded condition / neighbor latents so the dreamfix-side driver (`model_eval/flashdreams_backend.py`) can feed it directly. |
| Checkpoint loader | `state_dict_transform` for the merged ArtiFixer DMD safetensors produced by `dreamfix/scripts/merge_dcp_to_safetensors.py`. |

Cross-backend parity (captured single-scene `final_video`): **51.34 dB**
PSNR vs the dreamfix-native `ArtifixerKvCachePipeline` after the
fp32 AdaLN/norm/residual promotion and the no-op `finalize_kv_cache`
override on the FlashDreams transformer.

Set `ARTIFIXER_DMD_CHECKPOINT_PATH` to point at the merged ArtiFixer
DMD safetensors; alternatively set `ARTIFIXER_USE_BASE_WAN_WEIGHTS=1`
to fall back to vanilla Wan 2.1 1.3B HuggingFace weights (useful for
smoke-testing the recipe wiring before the merged safetensors are
available).

## Shipped slugs

| slug | description |
| --- | --- |
| `artifixer-dmd-wan2.1-t2v-1.3b` | ArtiFixer reconstruction-enhanced T2V (Wan 2.1 1.3B + opacity/camera/neighbor extensions, 4-step DMD). |

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

## Run

```bash
export HF_TOKEN=<your-hf-token>

uv run flashdreams-run artifixer-dmd-wan2.1-t2v-1.3b --help

uv run flashdreams-run artifixer-dmd-wan2.1-t2v-1.3b \
    --prompt "A photorealistic dolly-in shot of a modern living room." \
    --total-blocks 3
```

## Tests

```bash
uv run --extra dev pytest integrations/artifixer/tests
```
