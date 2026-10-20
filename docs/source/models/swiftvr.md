---
title: 'SwiftVR'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://h-oliday.github.io/SwiftVR/">Project page</a>
  <a class="fd-button" href="https://arxiv.org/abs/2606.09516">arXiv paper</a>
  <a class="fd-button" href="https://huggingface.co/H-oliday/SwiftVR">Checkpoint</a>
  <a class="fd-button" href="https://github.com/H-oliday/SwiftVR">Official code</a>
</div>

SwiftVR is a real-time, one-step streaming video-restoration model. It combines
mask-free shifted-window attention with a restoration-aware autoencoder for
causal chunk-wise inference. FlashDreams provides 2x and 4x post-processing
presets and the standalone `v2v-swiftvr` application.

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/model_clips/swiftvr/swiftvr-2x.avif" alt="" />
</div>

<figcaption class="tiny-figcaption">
  FlashDreams SwiftVR 2x output at 2560x1280. The input was the complete
  81-frame, 1280x640 Wan 2.2 sample on this site.
</figcaption>

## Quick Start



### V2V

```bash
uv sync --package flashdreams-swiftvr --inexact
uv run --no-sync flashdreams-run-v2 v2v-swiftvr \
  --mode mp4 --output-path artifacts/swiftvr-2x.mp4 --timeout unbound -- \
  --video-path docs/source/_static/model_clips/wan22/wan22-ti2v-5b.avif
```

<div class="fd-cta-row">
  <a class="fd-button" href="#v2v-presets">Demo Presets</a>
  <a class="fd-button" href="../demos/v2v.md#demo-arguments">Demo Arguments</a>
</div>

<hr>

## Demo Presets

<a id="v2v-presets"></a>

### V2V

| Preset | Description |
| --- | --- |
| `v2v-swiftvr` | SwiftVR 2x streaming video restoration. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/swiftvr">Integration Source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/swiftvr/config.py">Pipeline configurations</a>
</div>

- **GPU:** One CUDA-capable NVIDIA GPU.
- **Python:** 3.10 or newer.
- **PyTorch:** 2.9 or newer.
- **FFmpeg:** Required for MP4 input and output.

### Post-processing presets

Cam2V and Interactive Drive commands select these presets with
`--postprocess-preset PRESET` in their application arguments, after `--`:

| Preset | Description |
| --- | --- |
| `swiftvr-2x` | Startup-friendly eager 2x restoration. |
| `swiftvr-2x-compiled` | Compiled 2x restoration for long-running streams. |
| `swiftvr-4x` | Eager 4x restoration. |

<hr>

## Citation

If you use SwiftVR, please cite the original work:

```bibtex

@article{yan2026swiftvr,
  title={SwiftVR: Real-Time One-Step Generative Video Restoration},
  author={Yan, Jiaqi and Chen, Xiangyu and Zhong, Xinlin and Huang, Haibin and Zhang, Chi and Liu, Jie and Zhou, Jiantao and Li, Xuelong},
  journal={arXiv preprint arXiv:2606.09516},
  year={2026}
}
```
