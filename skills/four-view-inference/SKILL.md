---
name: four-view-inference
description: Run OmniDreams 4-view (multi-camera) HDMap→RGB inference with the flashdreams-run CLI. Use when asked to run, test, benchmark, or configure 4-view / multi-view / multi-camera OmniDreams inference, or to fetch the 4-view checkpoints. The authoritative guide is integrations/omnidreams/docs/four_view_inference.md — read it for setup, bring-your-own-data, memory, and quality; this skill is the quick operational pointer.
---

# Run 4-view OmniDreams inference

The `mv` (multi-view) runner rolls out all four OmniDreams cameras at once with
cross-view attention. **Full guide (authoritative — read it for anything not
covered here): `integrations/omnidreams/docs/four_view_inference.md`.** Keep
this skill thin; put details in the doc.

## Pick a checkpoint

| Runner slug | When |
|---|---|
| `omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae` | **default** — fastest, best image quality, 1 GPU |
| `omnidreams-mv-2steps-chunk4-loc8-vae-lighttae` | control variant (full Wan VAE encode + LightTAE decode) — full-VAE HDMap-control fidelity on **1 GPU**; image quality is a preview |
| `omnidreams-mv-2steps-chunk4-loc8-vae-vae` | same control encode with full Wan VAE decode (higher decode quality) — preview; **4-GPU only** |

The two `vae-*` variants share the `4view-vae-chunk4` checkpoint and differ only
in the decoder (LightTAE = 1 GPU; full Wan VAE = 4 GPU).

## Run (bundled example data)

```bash
export HF_TOKEN=<token>                       # access to omni-dreams-models
export OMNI_DREAMS_HF_ORG=<org>               # omit for the default nvidia org
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run flashdreams-run \
    omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae \
    --example-data True --total-blocks 20
```

Add `torchrun --nproc_per_node=4 --no-python` before `flashdreams-run` for the
4-GPU layout (one rank per camera). `--total-blocks` is AR chunks; the loop
auto-stops at the HDMap length.

## Cameras (canonical order)

Supply the four per-camera inputs (HDMap video + first frame) **space-separated**
in this canonical order:

1. `camera_cross_left_120fov`
2. `camera_cross_right_120fov`
3. `camera_front_tele_30fov`
4. `camera_front_wide_120fov`

`--camera-names` defaults to this list — omit it for canonical order. The model
selects each view's learned embedding **by name** (the four cameras map to
non-consecutive view indices), so pass `--camera-names` in matching order only
if you provide inputs in a different order.

## Key gotchas (see the doc for the rest)

- Env vars (`OMNI_DREAMS_HF_ORG`, `PYTORCH_CUDA_ALLOC_CONF`) are read at
  process/import start — export them **before** launching.
- HF `401/403/404` → check `HF_TOKEN` and that `OMNI_DREAMS_HF_ORG` matches your
  access; the file must exist under `<org>/omni-dreams-models/multi_view/four_camera/`.
- `pshuffle` and `vae-lighttae` fit one 80 GB H100; only `vae-vae`'s full Wan
  VAE decode needs the 4-GPU layout (`torchrun --nproc_per_node=4`). On a 1-GPU
  OOM, lower `--total-blocks` or go 4-GPU.
- After adding/renaming a runner, `uv sync` so entry-point discovery refreshes.
