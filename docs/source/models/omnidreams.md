---
title: 'NVIDIA OmniDreams'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/">Blog page</a>
  <a class="fd-button" href="https://arxiv.org/abs/2606.03159">Tech report</a>
  <a class="fd-button" href="https://huggingface.co/nvidia/omni-dreams-models/">Model page</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/omnidreams">Official code</a>
</div>

OmniDreams is an HDMap-conditioned streaming world model for driving
generation, with application configurations that balance visual fidelity and
runtime throughput.

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/model_clips/omnidreams/omnidreams-teaser.avif" alt="" />
</div>
<figcaption class="tiny-figcaption">
  Teaser video source:
  <a href="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/">OmniDreams project page</a>.
</figcaption>

## Quick Start



### Crazy Robotaxi

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
uv run --no-sync flashdreams-run-v2 crazy-robotaxi-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089
# Open http://127.0.0.1:8089/
```

<div class="fd-cta-row">
  <a class="fd-button" href="#crazy-robotaxi-presets">Demo Presets</a>
  <a class="fd-button" href="../demos/crazy_robotaxi.md#demo-arguments">Demo Arguments</a>
</div>

### Interactive Drive

```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
# Optional: preload assets and validate the first model block.
uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --preload-application

uv run --no-sync flashdreams-run-v2 interactive-drive-omnidreams \
  --mode webrtc --host 0.0.0.0 --port 8089
# Open http://127.0.0.1:8089/
```

<div class="fd-cta-row">
  <a class="fd-button" href="#interactive-drive-presets">Demo Presets</a>
  <a class="fd-button" href="../demos/interactive_drive.md#demo-arguments">Demo Arguments</a>
</div>

<hr>

## Demo Presets

<a id="crazy-robotaxi-presets"></a>

### Crazy Robotaxi

Fastest preset: `crazy-robotaxi-omnidreams-fast-perf`.

| Preset | Description |
| --- | --- |
| `crazy-robotaxi-omnidreams` | Standard configuration. |
| `crazy-robotaxi-omnidreams-optimized-gb300` | GB300-optimized attention. |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000` | RTX PRO 6000-optimized attention. |
| `crazy-robotaxi-omnidreams-perf` | Performance-tuned native acceleration. |
| `crazy-robotaxi-omnidreams-fast-perf` | Fastest preset; native FP8 LightVAE. |
| `crazy-robotaxi-omnidreams-rtx-5090` | 32 GB RTX 5090 configuration at `1168 x 640`. |
| `crazy-robotaxi-omnidreams-rtx-5090-fast` | Real-time RTX 5090 configuration with native FP8 LightVAE at `1024 x 560`. |
| `crazy-robotaxi-omnidreams-responsive` | Standard configuration with responsive model history. |
| `crazy-robotaxi-omnidreams-optimized-gb300-responsive` | GB300-optimized attention with responsive model history. |
| `crazy-robotaxi-omnidreams-optimized-rtx-pro-6000-responsive` | RTX PRO 6000-optimized attention with responsive model history. |
| `crazy-robotaxi-omnidreams-perf-responsive` | Performance schedule with responsive model history. |
| `crazy-robotaxi-omnidreams-fast-perf-responsive` | Native FP8 LightVAE with responsive model history. |

<a id="interactive-drive-presets"></a>

### Interactive Drive

Fastest preset: `interactive-drive-omnidreams-fast-perf`.

| Preset | Description |
| --- | --- |
| `interactive-drive-omnidreams` | Default single-view, two-step HDMap-conditioned configuration. |
| `interactive-drive-omnidreams-optimized-gb300` | GB300-optimized attention. |
| `interactive-drive-omnidreams-optimized-rtx-pro-6000` | RTX PRO 6000-optimized attention. |
| `interactive-drive-omnidreams-perf` | Performance-tuned native acceleration. |
| `interactive-drive-omnidreams-fast-perf` | Fastest preset; native FP8 LightVAE. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/omnidreams">Integration Source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/omnidreams/config.py">Pipeline configurations</a>
</div>

- **Minimum VRAM:** about 48 GB for the default Interactive Drive configuration.
- **PyTorch:** 2.11 or newer.
- **Python:** 3.10 through 3.12.

<hr>

## Performance (Outdated)

Single-view latency on NVIDIA GB300 at `704 x 1280`:

| Stage | 1x GPU | 2x GPU | 4x GPU | 8x GPU |
| --- | --- | --- | --- | --- |
| HDMap Encoder | 28 ms | 26 ms | 26 ms | 26 ms |
| Diffusion DiT | 84 ms | 71 ms | 49 ms | 47 ms |
| VAE Decoder | 6 ms | 5 ms | 5 ms | 5 ms |
| KV-cache Update | 42 ms | 34 ms | 23 ms | 22 ms |
| **Total** | **118 ms** | **102 ms** | **80 ms** | **78 ms** |
| **Effective FPS** | **68** | **78** | **100** | **103** |

<hr>

## Samples

<div class="model-media-grid zoomable">
  <div class="fd-card fd-model-card">
    <img class="fd-card-preview" src="../_static/model_clips/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.avif" alt="" />
    <figcaption class="tiny-figcaption">
      example_data_uuid: "239560dc-33d1-11ef-9720-00044bcbccac"
    </figcaption>
  </div>
  <div class="fd-card fd-model-card">
    <img class="fd-card-preview" src="../_static/model_clips/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-24b84744-4156-11ef-b27d-00044bf655de-pip.avif" alt="" />
    <figcaption class="tiny-figcaption">
      example_data_uuid: "24b84744-4156-11ef-b27d-00044bf655de"
    </figcaption>
  </div>
</div>

<hr>

## Citation

If you use OmniDreams, cite the original work:

```bibtex
@misc{nvidia2026omnidreams,
  title         = {{NVIDIA} {OmniDreams}: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation},
  author        = {Basant, Aarti and Kar, Amlan and Paschalidou, Despoina and Wei, Fangyin and Ferroni, Francesco and Garcia Cobo, Guillermo and Turki, Haithem and Ling, Huan and Seo, Jaewoo and Lucas, James and Wu, Jay Zhangjie and Wang, Jialiang and Lorraine, Jonathan and Gao, Jun and He, Kai and Tothova, Katarina and Xie, Kevin and Tyszkiewicz, Micha{\l} and Wu, Qi and de Lutio, Riccardo and Li, Ruilong and Fidler, Sanja and Kim, Seung Wook and Shen, Tianchang and Cao, Tianshi and Pfaff, Tobias and Lew, William and Wu, Xindi and Ren, Xuanchi and Lu, Yifan and Zhang, Yuxuan and Gojcic, Zan and Wang, Zian},
  year          = {2026},
  eprint        = {2606.03159},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2606.03159},
  url           = {https://arxiv.org/abs/2606.03159},
}
```
