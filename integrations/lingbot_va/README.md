<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot-VA Robotwin I2AV

This workspace package implements the LingBot-VA dual video/action model. The
V2 application adapter lives in `integrations_v2/lingbot_va`; model code here
has no CLI, MP4, metrics-file, or action-file ownership.

The port is based on FlashDreams PR #312, with its CFG cache ownership,
checkpoint loading, configuration propagation, lifecycle, and output contracts
reworked for the V2 API. The original unverified 2.3x/1.48x performance claims
have been removed; only measurements produced by the checked-in implementation
and matched parity harness are reported.

## Install

From the repository root:

```bash
uv sync --project integrations_v2/lingbot_va
```

The tested model dependency window is Diffusers 0.38.x and Transformers 5.x.
The engine uses private Wan VAE streaming fields, so widening the Diffusers
range requires a real-model retest.

## Run through V2

The reproducible default uses the official checkpoint revision
`8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`:

```bash
uv run --project integrations_v2/lingbot_va flashdreams-run-v2 \
    lingbot-va-robotwin-i2av \
    --mode mp4 \
    --output-path outputs/lingbot_va/demo.mp4 \
    --stats-path outputs/lingbot_va/metrics.json \
    --tensor-artifact-dir outputs/lingbot_va \
    -- \
    --checkpoint-root robbyant/lingbot-va-posttrain-robotwin \
    --checkpoint-revision 8c9dea8abbc5c91cc9e18bc3264b8915083bbe70 \
    --input-image-dir assets/example_data/lingbot-va/robotwin \
    --num-chunks 10
```

Use `--no-compile` for correctness debugging and `--enable-offload` when GPU
memory is constrained. `flashdreams-run-v2 lingbot-va-robotwin-i2av -- --help`
lists every effective model override.

### Checkpoint modes

`--checkpoint-root` accepts either:

- a local snapshot root containing `transformer/`, `vae/`, `text_encoder/`,
  and `tokenizer/`; or
- a Hugging Face repository ID, optionally pinned with
  `--checkpoint-revision`.

Existing paths are always treated as local. Prefix a not-yet-created relative
local path with `./` so it fails as a local path rather than being interpreted
as a repository ID. All `from_pretrained` calls use a resolved root plus an
explicit subfolder and `local_files_only=True`.

### Three-camera inputs

The input directory must contain:

- `observation.images.cam_high.png`
- `observation.images.cam_left_wrist.png`
- `observation.images.cam_right_wrist.png`

The high camera is encoded at 256x320. Each wrist camera is encoded at 128x160,
and their latents form the upper bar of the upstream Robotwin T layout. The
repository defaults are the official upstream example images; their source and
hashes are recorded beside the assets.

### Outputs

One V2 model step returns the complete rollout:

- video: float tensor `[time, 3, 256, 320]`, range `[-1, 1]`, 10 FPS;
- `actions` artifact: float tensor `[step, channel]`;
- timing and peak-allocation metrics.

Each chunk produces 2 latent frames and 32 action steps. Wan's temporal decoder
turns `2N` accumulated latent frames into `8N - 3` pixel frames. The decoded
T-layout is cropped to its 256x320 high-camera view for the V2 video channel;
the action artifact has 16 selected Robotwin channels, ordered by channel IDs
`0..6, 28, 7..13, 29`. MP4, JSON, and NumPy serialization belong to generic V2
runtime sinks; the engine itself performs no output I/O.

## Lifecycle and limitations

Video decoding needs the DiT, text encoder, and KV state released first. A
session therefore owns a destructive, one-run engine. Reset closes that engine
and lazily creates a new one. The initial implementation honestly returns one
long model step after all chunks are generated and decoded; it does not claim
per-chunk interactive streaming.

Only one GPU is currently supported. Multi-GPU/FSDP execution and live
RoboTwin control are outside this I2AV adapter. See `DESIGN.md` for ownership,
state transitions, and failure semantics. See `GPU_EVIDENCE.md` for exact
checkpoint parity bounds, real multi-chunk artifacts, memory measurements, and
the opt-in reproduction commands.

## Real-model verification

The checked-in GPU test is opt-in because the checkpoint is about 23 GiB:

```bash
LINGBOT_VA_REAL_MODEL_RUN=1 uv run --no-sync pytest \
    integrations_v2/lingbot_va -m ci_gpu -s
```

Set `LINGBOT_VA_CHECKPOINT_ROOT` to reuse a resolved local snapshot. The
separate `LINGBOT_VA_REAL_MODEL_COMPILE_RUN=1` gate exercises the cold
`torch.compile` path. The upstream comparison harness and accepted numerical
bounds are documented in `GPU_EVIDENCE.md`.

## Provenance

- Source architecture/inference reference:
  `robbyant/lingbot-va@7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`.
- Official Robotwin checkpoint:
  `robbyant/lingbot-va-posttrain-robotwin@8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`.
- Initial FlashDreams draft source: PR #312 at
  `f98cae4a18ddf6c189a6cfa2099265d6d570e337`.
