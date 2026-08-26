<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot-VA Robotwin I2AV

This workspace package implements the LingBot-VA dual video/action model. The
V2 application adapter lives in `integrations_v2/lingbot_va`; model code here
has no CLI, MP4, metrics-file, or action-file ownership.

The implementation transplants the useful model work from draft PR #312 and is
otherwise self-contained. CFG cache ownership, checkpoint loading, configuration
propagation, lifecycle, and output contracts were corrected for the V2 API. The
draft performance claims are intentionally omitted; only matched measurements
from the checked-in implementation are recorded below.

## Install and run

From the repository root:

```bash
uv sync --project integrations_v2/lingbot_va

uv run --project integrations_v2/lingbot_va flashdreams-run-v2 \
    lingbot-va-robotwin-i2av \
    --mode mp4 \
    --output-path outputs/lingbot_va/demo.mp4 \
    --stats-path outputs/lingbot_va/metrics.json \
    --tensor-artifact-dir outputs/lingbot_va \
    -- \
    --checkpoint-root robbyant/lingbot-va-posttrain-robotwin \
    --checkpoint-revision 8c9dea8abbc5c91cc9e18bc3264b8915083bbe70 \
    --input-image-dir /path/to/robotwin-images \
    --num-chunks 10
```

`--input-image-dir` is required; this repository does not bundle example
images. Use `--no-compile` for correctness debugging and `--enable-offload`
when GPU memory is constrained.
`flashdreams-run-v2 lingbot-va-robotwin-i2av -- --help` lists every model
override.

The tested dependency window is Diffusers 0.38.x and Transformers 5.x. The
engine uses private Wan VAE streaming fields, so widening the Diffusers range
requires a real-model retest.

## Inputs and checkpoints

The input directory must contain exactly named Robotwin camera files:

- `observation.images.cam_high.png`
- `observation.images.cam_left_wrist.png`
- `observation.images.cam_right_wrist.png`

The high camera is encoded at 256x320. Each wrist camera is encoded at 128x160,
and their latents form the upper bar of the upstream Robotwin T layout.

The measured runs used the unmodified examples from
`robbyant/lingbot-va@7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`,
under `example/robotwin/`. Their upstream introduction commit is
`5ed0eb32046b34fe5c14f929d81e87ab6ebe02ef`.

| File | SHA-256 |
| --- | --- |
| `observation.images.cam_high.png` | `78cab76d394114ba912f882ac9b00ddc017f98c946482cb9267c87de82486b72` |
| `observation.images.cam_left_wrist.png` | `fbe55b713e1b3d4505fda6be3b00132213ee1725a78cb357ed2bbd5b3a3a7a93` |
| `observation.images.cam_right_wrist.png` | `b9e6821b38073567232f6dd8f6389b123d09d3d1a70758c27d388e547e228249` |

`--checkpoint-root` accepts either a local snapshot containing
`transformer/`, `vae/`, `text_encoder/`, and `tokenizer/`, or a Hugging
Face repository ID optionally pinned with `--checkpoint-revision`. Existing
paths are always local. Prefix a missing relative local path with `./` so it
fails locally rather than being interpreted as a repository ID. Component
loads use one resolved root, explicit subfolders, and `local_files_only=True`.

## V2 outputs

One model step returns the complete rollout:

- video: float tensor `[time, 3, 256, 320]`, range `[-1, 1]`, 10 FPS;
- `actions`: float tensor artifact `[step, channel]`;
- timing and peak-allocation metrics.

Each chunk produces two latent frames and 32 action steps. Wan temporal decoding
turns `2N` accumulated latent frames into `8N - 3` pixel frames. The decoded
T layout is cropped to its 256x320 high-camera view. Actions contain 16 selected
Robotwin channels in the order `0..6, 28, 7..13, 29`.

MP4, JSON, and NumPy serialization belong to generic V2 runtime sinks. The
adapter declares `actions[step, channel]` once and attaches the tensor to its
single model-result channel. Backpressure and presentation policies selected by
the runtime are preserved; the fixed video layout, dimensions, rate, and
artifact schema are validated.

## Design and lifecycle

Each session owns one `LingbotVAEngine`; the application owns only immutable
configuration and an engine factory. The successful engine state machine is
`NEW -> RUNNING -> FINISHED -> CLOSED`. Calling `run` outside `NEW` is an
error. Any inference failure triggers cleanup and transitions to `CLOSED`.

VAE decoding cannot fit beside the complete DiT, text encoder, and KV footprint
on the supported capacity path. Generation therefore releases denoising state
before moving the VAE to the decode device. This transition is destructive, so
the adapter emits one honest long model step rather than claiming per-chunk
interactive presentation. Reset closes the current engine; the next step lazily
creates a fresh one. Close is idempotent, and cleanup errors do not replace an
earlier inference failure.

| Phase | GPU/active | CPU/host | Released at boundary |
| --- | --- | --- | --- |
| load | DiT; optionally VAE/T5 | tokenizer; offloaded components | partial state on failure |
| encode | T5 and VAE as needed | three input PNGs | prompt/observation temporaries |
| denoise | DiT, CFG caches, latent/action state | completed chunks | per-step temporaries |
| teardown | VAE after transfer | DiT/T5/tokenizer references | KV and denoising state |
| decode | VAE plus one decoded frame | output frames/actions | VAE cache and GPU frames |
| finished | none | returned video/actions/metrics | all model components |

The default CFG scales are video 5 and action 1. Conditional and unconditional
branches own distinct video KV and both advance whenever CFG is active. Each
action denoise pass attends to committed prior chunks plus its matching branch's
current video KV before adding fresh action KV. The cache attention window is 72,
matching the pinned upstream Robotwin configuration.

Deliberate exclusions are the V1 runner, application-owned output files,
application-owned model components, multi-GPU/FSDP claims, live RoboTwin
control, unmeasured speedup claims, and root CUDA/Torch policy changes.

## Validation evidence

The evidence below was produced on 2026-08-25 using an NVIDIA RTX PRO 6000
Blackwell Workstation Edition (97,887 MiB), driver 595.84, PyTorch 2.12.1+cu130,
CUDA 13.0, BF16, one CUDA device, and seed 42. It is validation evidence, not a
general performance claim.

The official checkpoint revision
`8c9dea8abbc5c91cc9e18bc3264b8915083bbe70` contained 841 transformer entries.
Two obsolete `patch_embedding.*` entries are deliberately dropped. All 839
remaining keys map bijectively to the 839 native entries, producing a
5,088,872,670-parameter transformer under strict loading.

### Upstream flow parity

`tools/compare_upstream.py` loads pinned upstream and native transformers
sequentially, then drives the same first-chunk video and action tensors through
the same cache lifecycle. Acceptance gates are maximum absolute error <= 0.07
and mean absolute error <= 0.012 for each stream.

| Native mode | Stream | Maximum absolute error | Mean absolute error | RMS error |
| --- | --- | ---: | ---: | ---: |
| eager | video | 0.04296875 | 0.00751040 | 0.00951632 |
| eager | action | 0.06250000 | 0.00875314 | 0.01267146 |
| compiled | video | 0.05468750 | 0.00970979 | 0.01229867 |
| compiled | action | 0.06250000 | 0.01116651 | 0.01560142 |

The comparison exposed an action-attention defect: current-video KV was passed
into the native action block loop but omitted from attention. Before the fix,
action maximum/mean errors were 1.015625/0.203186. The table records the fixed
behavior, now covered by CPU cache regressions.

```bash
PYTHONPATH=flashdreams:integrations/lingbot_va \
python integrations/lingbot_va/tools/compare_upstream.py \
    --checkpoint-root /path/to/resolved/snapshot \
    --upstream-root /path/to/robbyant-lingbot-va-7c6ffa9 \
    --compile-native \
    --maximum-video-error 0.07 --mean-video-error 0.012 \
    --maximum-action-error 0.07 --mean-action-error 0.012
```

### Real multi-chunk V2 run

Matched resident and offloaded two-chunk runs used default CFG, 25 video steps,
50 action steps, and no compilation. Both produced finite BF16 video
`[13, 3, 256, 320]`, finite float32 actions `[64, 16]`, distinct chunk means
0.179792 and 0.333868, a valid 13-frame 320x256 10 FPS H.264 MP4, and
byte-identical resident/offload outputs.

| Artifact | SHA-256 |
| --- | --- |
| `demo.mp4` | `df0c193137a673f4f8d6b2372b4bf7afc01a9937c4280b7fa5a51912b5e93c1a` |
| `actions.npy` | `463b307b667c1ca13a47bbbc5a17f68604621dfe3c3a10fc5860077216928d95` |

| Mode | Prompt | Observation | Denoise | Decode | Total | Peak allocation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| resident | 0.240 s | 0.221 s | 4.735 s | 0.260 s | 33.220 s | 40.35 GiB |
| offload | 5.055 s | 0.776 s | 4.744 s | 0.429 s | 34.009 s | 37.07 GiB |

After close, only the process CUDA allocator context remained (about 0.031
GiB). A separate complete compiled one-chunk run returned finite
`[5, 3, 256, 320]` video and `[32, 16]` actions. Cold compilation is excluded
from the timing table and no speedup claim is made.

Run the opt-in production test with explicit input images:

```bash
LINGBOT_VA_REAL_MODEL_RUN=1 \
LINGBOT_VA_INPUT_DIR=/path/to/robotwin-images \
uv run --no-sync pytest integrations_v2/lingbot_va -m ci_gpu -s
```

Set `LINGBOT_VA_CHECKPOINT_ROOT` to reuse a local snapshot.
`LINGBOT_VA_REAL_MODEL_COMPILE_RUN=1` separately enables the cold compile test.

### Final stacked-PR revalidation

After the review fixes, the same pinned checkpoint and input hashes were run
through the final stacked code with two chunks, default CFG, offload, and no
compilation:

- GPU test: 1 passed in 34.02 s;
- MP4: H.264, 320x256, 10 FPS, 13 frames, SHA-256
  `15bcdc4307e080218255e83946c2c2e5dbc30f3b7acd26c8925167017234e586`;
- actions: float32 `[64, 16]`, finite, distinct chunks, SHA-256
  `463b307b667c1ca13a47bbbc5a17f68604621dfe3c3a10fc5860077216928d95`;
- peak allocation: 39,804,415,488 bytes (37.07 GiB).

The action hash exactly matches the earlier resident/offload parity run.

## Remaining limits and provenance

The current implementation supports one GPU, one complete deferred-decode
rollout, and Diffusers 0.38 private VAE streaming state. It makes no FSDP,
context-parallel, or per-chunk presentation claim.

- Source architecture/inference:
  `robbyant/lingbot-va@7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`.
- Official checkpoint:
  `robbyant/lingbot-va-posttrain-robotwin@8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`.
- Initial draft source:
  FlashDreams PR #312 at `f98cae4a18ddf6c189a6cfa2099265d6d570e337`.
