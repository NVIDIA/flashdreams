---
title: 'Causal-Forcing'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://thu-ml.github.io/CausalForcing.github.io/">Project page</a>
  <a class="fd-button" href="https://arxiv.org/abs/2602.02214">arXiv paper</a>
  <a class="fd-button" href="https://github.com/thu-ml/Causal-Forcing">Official code</a>
</div>

Causal-Forcing uses Causal ODE or Causal Consistency Distillation to drive
asymmetric DMD as a theoretically correct initialization for real-time
interactive video generation.

![Causal-Forcing overview figure.](https://thu-ml.github.io/CausalForcing.github.io/images/overview.png)

<figcaption class="tiny-figcaption">
  Teaser image source:
  <a href="https://thu-ml.github.io/CausalForcing.github.io/">Causal-Forcing project page</a>.
</figcaption>

## Quick Start



### T2V

```bash
uv sync --package flashdreams-causal-forcing --inexact
uv run --no-sync flashdreams-run-v2 \
  t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise \
  --output-path artifacts/causal-forcing.mp4 --timeout unbound -- \
  --prompt "A cat surfing" --total-blocks 7
```

- [Demo presets](#t2v-presets)
- [Demo arguments](../demos/t2v.md#demo-arguments)

<hr>

## Demo Presets

<a id="t2v-presets"></a>

### T2V

| Preset | Description |
| --- | --- |
| `t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise` | Chunkwise Wan 2.1 1.3B T2V (`len_t=3`). |
| `t2v-causal-forcing-wan2.1-t2v-1.3b-framewise` | Framewise Wan 2.1 1.3B T2V (`len_t=1`). |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/causal_forcing">Integration source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/causal_forcing/config.py">Pipeline configurations</a>
</div>

- **Minimum VRAM:** about 24 GB.
- **PyTorch:** 2.9 or newer.

<hr>

## Samples

<div class="model-video-grid zoomable">
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_forcing/causal-forcing-wan2.1-t2v-1.3b-framewise.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption class="tiny-figcaption">
      prompt: "A cinematic closeup and detailed portrait of a reindeer standing in a snowy forest at sunset. The lighting is gorgeous and soft, with a golden backlight creating a warm and dreamy effect. Soft bokeh and lens flares add a magical touch, enhancing the cinematic quality of the image. The reindeer has a gentle expression, its fur glistening in the fading light. The background features a serene snowy landscape with tall trees silhouetted against the orange and pink hues of the setting sun. The color grade is rich and magical, capturing the essence of a winter wonderland at twilight. A close-up shot from a slightly elevated angle."
    </figcaption>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_forcing/causal-forcing-wan2.1-i2v-1.3b-framewise.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption class="tiny-figcaption">
      prompt: "A cinematic closeup and detailed portrait of a reindeer standing in a snowy forest at sunset. The lighting is gorgeous and soft, with a golden backlight creating a warm and dreamy effect. Soft bokeh and lens flares add a magical touch, enhancing the cinematic quality of the image. The reindeer has a gentle expression, its fur glistening in the fading light. The background features a serene snowy landscape with tall trees silhouetted against the orange and pink hues of the setting sun. The color grade is rich and magical, capturing the essence of a winter wonderland at twilight. A close-up shot from a slightly elevated angle."
      <br/>
      image: https://raw.githubusercontent.com/thu-ml/Causal-Forcing/refs/heads/main/prompts/i2v/26-15/000001.png
    </figcaption>
  </div>
</div>

<hr>

## Citation

If you use Causal-Forcing, please cite the original work:

```bibtex

@article{zhu2026causal,
  title={Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation},
  author={Zhu, Hongzhou and Zhao, Min and He, Guande and Su, Hang and Li, Chongxuan and Zhu, Jun},
  journal={arXiv preprint arXiv:2602.02214},
  year={2026}
}
```
