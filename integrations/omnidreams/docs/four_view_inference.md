<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Run 4-view OmniDreams inference

Streaming HDMap→RGB rollout across the **four OmniDreams cameras** at once,
with cross-view attention so the views stay geometrically consistent. This is
the multi-view (`mv`) counterpart of the single-view (`sv`) runner; the only
structural difference is `num_views=4` and the per-camera inputs.

Two 4-view checkpoints ship (`pshuffle` and `vae`); the `vae` checkpoint is
offered with two decoders. Pick by need:

| Slug (runner name) | Encode + decode | Speed / GPUs | Use it for |
|---|---|---|---|
| `omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae` | pixel-shuffle encode + LightTAE decode | **fastest**, 1 GPU | Default. Best image quality today. |
| `omnidreams-mv-2steps-chunk4-loc8-vae-lighttae` | full Wan VAE encode + LightTAE decode | 1 GPU | Full-VAE HDMap-control fidelity **on a single GPU** (preview). |
| `omnidreams-mv-2steps-chunk4-loc8-vae-vae` | full Wan VAE encode + decode | **4-GPU only** | Same control fidelity with the higher-quality full-VAE decode (preview — see [§9](#9-image-quality-pshuffle-vs-vae)). |

All are 2-step distilled, chunk4 (`len_t=4`, `window_size_t=8`), 720p,
30 FPS, and render each view at native **704×1280**. The two `vae-*` variants
share the `4view-vae-chunk4` checkpoint and differ only in the decoder
(LightTAE vs full Wan VAE).

---

## 1. Checkpoints

Both 4-view checkpoints are published on Hugging Face under
`<org>/omni-dreams-models` and download automatically on first run:

| Slug | HF file (under `omni-dreams-models`) |
|---|---|
| `4view-pshuffle-chunk4` | `multi_view/four_camera/32n_cosmos_v2_2b_SF_4view_res720p_fps30_chunk4_i2v_hdmap_pixel_shuffle_loc8st2_gcp.pt` |
| `4view-vae-chunk4` | `multi_view/four_camera/32n_cosmos_v2_2b_SF_4view_res720p_fps30_chunk4_i2v_hdmap_vae_encoding_loc8st2_gcp.pt` |

`nvidia` is the default Hugging Face org. If you have early access to
experimental checkpoints under a different authorized org, select it by
setting `OMNI_DREAMS_HF_ORG`:

```bash
export OMNI_DREAMS_HF_ORG=<your-authorized-org>
```

The org is read when `omnidreams.config` is imported, so set it **before**
launching.

## 2. Setup

```bash
export HF_TOKEN=<your-hf-token>            # access to the models repo
export OMNI_DREAMS_HF_ORG=<your-org>       # omit to use the default nvidia org
```

On first run the checkpoint (~5 GB), the Wan VAE / LightTAE autoencoders, and
the Cosmos-Reason1 text encoder (~14 GB) download to the Hugging Face cache.

**Prerequisite — `ffmpeg` on `PATH`.** The runner reads the per-camera HDMap
`.mp4` inputs and writes the `.mp4` output by shelling out to the `ffmpeg` CLI
(via `mediapy`); `uv sync` does **not** provide an `ffmpeg` executable, so
install one with your system package manager — there is no in-process fallback:

```bash
sudo apt install ffmpeg      # Debian / Ubuntu; use your distro's equivalent elsewhere
```

## 3. Cameras

Provide the four per-camera inputs (HDMap video + first frame) in this
**canonical order**:

1. `camera_cross_left_120fov`
2. `camera_cross_right_120fov`
3. `camera_front_tele_30fov`
4. `camera_front_wide_120fov`

You normally don't pass camera names at all — they default to this canonical
list. The names matter because the model selects each view's **learned
embedding by camera name**, not by position: the four cameras map to specific,
non-consecutive view indices the model was trained with, so the name (not the
slot) tells the model which physical camera a stream is. `--camera-names` is
therefore only needed if you supply the inputs in a **different** order — then
pass the names in that same order so each name still pairs with its input.

## 4. Quickstart (bundled example data)

`--example-data True` downloads one public 4-view clip (HDMap + RGB per camera)
from the NuRec sample set
(`nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`, rev `26.01`) and fills the four
per-camera inputs in canonical order. Pick a different clip with
`--example-data-uuid <clip>`.

**4-GPU** (one rank per camera — recommended when you have the GPUs):

```bash
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae \
    --example-data True --total-blocks 20
```

**1-GPU** (all four views on one 80 GB H100):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run flashdreams-run \
    omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae \
    --example-data True --total-blocks 20
```

For the full-VAE-encode control variants, swap the slug: `...-vae-lighttae`
(LightTAE decode) runs on the **1-GPU** command above; `...-vae-vae` (full Wan
VAE decode) needs the **4-GPU** command, since its 4-view full-VAE decode does
not fit one GPU.

## 5. Bring your own data

Pass four HDMap videos and four first frames (image **or** video — frame 0 is
used) in canonical camera order ([§3](#3-cameras)). Each multi-value argument
takes its values **space-separated** (one token per camera):

```bash
uv run flashdreams-run \
    omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae \
    --hdmap-video-paths  cross_left_hdmap.mp4 cross_right_hdmap.mp4 front_tele_hdmap.mp4 front_wide_hdmap.mp4 \
    --first-frame-paths  cross_left_rgb.mp4 cross_right_rgb.mp4 front_tele_rgb.mp4 front_wide_rgb.mp4 \
    --total-blocks 20
```

The path arguments are **file names** (space-separated, four tokens each). You
don't need `--camera-names` when the inputs are in canonical order — it defaults
to the canonical names; pass it (in matching order) only if you supply the
inputs in a different order. Prompts are optional: `--prompt "..."` applies one
text prompt to every view, or
`--prompts "left view text" "right view text" "front-tele text" "front-wide text"`
sets a different text prompt per view (four space-separated, individually quoted
strings — quote each so embedded commas/spaces stay within one prompt). The
defaults render an urban daytime street scene.

## 6. Rollout length (`--total-blocks`)

Each block is one autoregressive chunk (`len_t=4` latent frames). The loop
**auto-stops** when the HDMap video is consumed, so over-requesting is safe.

| `--total-blocks` | Output |
|---|---|
| 12 | quick check |
| 20 | short clip |
| 76 | ≈ full ~19.6 s clip @ 30 FPS (589 frames) |

**Performance.** Each generation uses one random seed; the first generation in
a process also pays a one-time `torch.compile` autotune (a few minutes) plus
the model build/checkpoint load (~40–130 s), after which each subsequent
generation runs at steady state. On one 80 GB H100 at `--total-blocks 12`
(189 frames), steady-state generation is roughly **1 minute** for `pshuffle`
and **~1.3 minutes** for `vae-lighttae`. `vae-vae` (full Wan VAE decode) runs on
the 4-GPU layout.

## 7. Memory

`pshuffle` and `vae-lighttae` both fit a **single 80 GB H100**: CUDA-graph
capture is disabled on the multi-view DiT (and, for `vae-lighttae`, on its
Wan-VAE HDMap encoder + LightTAE decoder) while `torch.compile` stays on. At
launch, add:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`vae-vae` instead decodes with the full Wan VAE, whose 4-view decode does
**not** fit one 80 GB H100, so it requires the **4-GPU** (one-rank-per-camera)
context-parallel layout, which splits the decode across ranks. For the 1-GPU
variants, if you still hit CUDA out-of-memory, lower `--total-blocks` or move
to the 4-GPU layout.

## 8. Outputs

The runner writes the generated views and per-step timing/stats under
`--output-dir` (default `outputs/`).

## 9. Image quality: `pshuffle` vs `vae`

`pshuffle` is the **recommended default** and currently has the better image
quality. The two `vae-*` variants are a **preview**: they use the full Wan VAE
to encode the HDMap, giving higher-fidelity map conditioning (useful for
control / closed-loop simulation), but their image quality is below `pshuffle`
on the current checkpoint and is still being improved. They share the
`4view-vae-chunk4` checkpoint and differ only in the decoder:

- **`vae-lighttae`** — LightTAE decode; **fits a single GPU**; faster.
- **`vae-vae`** — full Wan VAE decode; **higher-quality decode**, but needs the
  4-GPU layout.

Use `pshuffle` when visual quality is the priority (and to run on one GPU);
`vae-lighttae` for control fidelity on a single GPU; `vae-vae` when you want the
full-VAE decode quality and have four GPUs.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Program 'ffmpeg' is not found` | Install the `ffmpeg` CLI (`sudo apt install ffmpeg`); see [§2](#2-setup). `mediapy` shells out to it to read/write the `.mp4` videos and `uv sync` does not provide it. |
| `401/403` from Hugging Face | `HF_TOKEN` unset or lacks access; if using a non-default org, set `OMNI_DREAMS_HF_ORG` to your authorized org. |
| `404` on a 4-view file | Confirm the file exists under `<org>/omni-dreams-models/multi_view/four_camera/`; the org must match your access. |
| Example clip not found | Pick a clip from the NuRec `sample_set/26.01_release` tree and pass `--example-data-uuid <clip>`, or supply explicit `--hdmap-video-paths` / `--first-frame-paths`. |
| CUDA out-of-memory | Export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; lower `--total-blocks`; or use the 4-GPU layout. |
| Wrong / swapped views | The four inputs and `--camera-names` must be positionally aligned in the canonical order in [§3](#3-cameras). |

## See also

- `integrations/omnidreams/README.md` — package overview + single-view / serving.
- `flashdreams-run --help` and `flashdreams-run <slug> --help` — every
  overridable field.
