---
title: 'Waypoint 1.5'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://huggingface.co/Overworld/Waypoint-1.5-1B">Checkpoint and upstream model card</a>
  <a class="fd-button" href="https://github.com/Overworldai/world_engine">Official inference code</a>
  <a class="fd-button" href="https://github.com/Overworldai/Biome">Official desktop client</a>
</div>

Waypoint-1.5-1B is Overworld's dense, autoregressive interactive video world
model. FlashDreams integrates the published BF16 checkpoint as an
image-established, keyboard/mouse-controlled V2 application with deterministic
per-action metrics and MP4, WebRTC, or native-window presentation.

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://huggingface.co/Overworld/Waypoint-1.5-1B/resolve/main/assets/wp_1.5.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<figcaption class="tiny-figcaption">
  Upstream Waypoint 1.5 teaser from the
  <a href="https://huggingface.co/Overworld/Waypoint-1.5-1B">Overworld model card</a>;
  this is not a FlashDreams benchmark artifact.
</figcaption>

<div class="transparent-section" markdown>
## Quick Start



### Action2V

```bash
uv sync --package flashdreams-waypoint --inexact
uv run --no-sync flashdreams-run-v2 action2v-waypoint-1-5-1b \
  --mode webrtc --host 127.0.0.1 --port 8766 -- \
  --example-data --seed 464
```

- [Demo presets](#action2v-presets)
- [Demo arguments](../demos/action2v.md#demo-arguments)

</div>

<div class="transparent-section" markdown>
## Demo Presets

<a id="action2v-presets"></a>

### Action2V

| Preset | Description |
| --- | --- |
| `action2v-waypoint-1-5-1b` | Waypoint 1.5 1B with image-established keyboard and mouse control. |

</div>

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/waypoint">Integration source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/waypoint/config.py">Pipeline configurations</a>
</div>

- **GPU:** A CUDA-capable NVIDIA GPU with BF16 and PyTorch FlexAttention support.
- **Validated hardware:** One NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
  The measured PyTorch peak allocation was 6.061 GiB, but this is not a
  minimum-VRAM guarantee and excludes non-PyTorch process memory.
- **Model downloads:** The 3.72 GB BF16 Waypoint safetensors file and separate
  Overworld-Models/taehv1_5 checkpoint are cached on first run.

## Performance (Outdated)

The final FlashDreams path was measured on 2026-08-26 using an RTX PRO 6000
Blackwell Workstation Edition (96 GiB), driver 595.84, PyTorch 2.12.1+cu130,
CUDA 13.0, BF16 weights, the pinned validation first frame and control timeline, seed 464,
native 1024x512 output, four denoise evaluations, and synchronous profiling.
Actions 1-19 were warmup; actions 20-40 were the steady-state sample.

| Measurement | Result |
| --- | --- |
| Mean action latency | 68.791 ms per four generated frames |
| Median action latency | 68.887 ms |
| p90 action latency | 69.737 ms |
| Throughput derived from mean latency | 58.15 generated RGB frames/s |
| Peak PyTorch CUDA allocation | 6.061 GiB |
| Encoded output | 164 frames: 4 seed + 40 actions x 4 frames |

The declared 60 FPS is presentation timing, not a throughput guarantee.
Overworld separately reports 56 FPS for its unquantized runtime on an RTX 5090;
that result was not reproduced here and is not directly comparable across
different GPU, runtime, and presentation stacks.

### Parity and rollout validation

FlashDreams loaded the same BF16 checkpoint and matched the pinned official
world_engine implementation through one complete controlled action. The final
comparison includes the four-step Euler solve and both cache commits:

| Comparison | Mean absolute error | Max absolute error | Cosine similarity |
| --- | --- | --- | --- |
| Seed cache flow | 0.010793 | 0.218750 | 0.999586 |
| Clean latent after four steps | 0.005682 | 0.033691 | 0.999869 |
| Final cache flow | 0.010511 | 0.265625 | 0.999487 |

The residual is consistent with BF16 execution through different compiled
kernel boundaries. Review optimizations retained a byte-identical 40-action
MP4. Additional inference completed 15 distinct scenes at 40 actions each and
one 118-action rollout: 718 generated actions and 2,936 decoded frames in total,
with complete finite metrics and exact frame accounting. This demonstrates
execution, cache longevity, and scene coverage; it is not a qualitative
gameplay or physical-accuracy score.

[Complete validation record](https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/waypoint/VALIDATION.md)

## Citation

If you use Waypoint-1.5, cite the original work:

```bibtex
@misc{rajpal2026waypoint,
  title         = {Waypoint-1.5: A Real-Time Video World Model for Consumer Hardware},
  author        = {Rajpal, Rajit and Matiana, Shahbuland and {Liew Wei Pyn} and Agarwal, Anmol and Craig, Ryan and Lapp, Andrew and Hunsur, Mithun and BuGhanem, Sami and Fox, Scottie and Sanders, Aaron and Poole, Carson and Park, Irene and Rossi, David and Frazier, Spencer and Castricato, Louis},
  year          = {2026},
  eprint        = {2609.37107},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2609.37107},
  url           = {https://arxiv.org/abs/2609.37107},
}
```
