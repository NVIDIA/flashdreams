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
| 1 | Recipe scaffold + AR/scheduler knobs match dreamfix stage-3 DMD | this commit |
| 2 | Per-block opacity + camera MLPs, neighbor cross-attn, PRoPE | upcoming |
| 3 | Opacity-weighted latent mixing + self-forcing renoise loop | upcoming |
| 4 | dreamfix-format conditioning surface (rgb_rendered, opacity, neighbors) | upcoming |
| 5 | `state_dict_transform` for the merged ArtiFixer DMD safetensors | upcoming |

Phase 1 loads vanilla Wan 2.1 1.3B base weights from HuggingFace, so the
output is a plain T2V video. The recipe is wired up but the ArtiFixer
architectural extensions land in later commits.

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
