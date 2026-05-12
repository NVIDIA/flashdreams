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

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 1 | Recipe scaffold + AR/scheduler knobs match dreamfix stage-3 DMD | done |
| 2 | Per-block opacity + camera MLPs, neighbor cross-attn, PRoPE (parity-tested vs dreamfix) | done |
| 3 | Opacity-weighted latent mixing + self-forcing renoise loop | done |
| 4 | dreamfix-format conditioning surface (rgb_rendered, opacity, neighbors) | upcoming |
| 5 | `state_dict_transform` for the merged ArtiFixer DMD safetensors | done |

By default the recipe loads the merged ArtiFixer DMD safetensors from
`ARTIFIXER_DMD_CHECKPOINT_PATH` (defaults to a `/lustre` path that
matches the dreamfix repo's `merged_checkpoints/`); set
`ARTIFIXER_USE_BASE_WAN_WEIGHTS=1` to fall back to vanilla Wan 2.1 1.3B
HuggingFace weights (useful for smoke-testing the recipe wiring before
the merged safetensors are available).

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
