---
title: 'HY-WorldPlay'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://3d-models.hunyuan.tencent.com/world/" target="_blank" rel="noopener noreferrer">Project page</a>
  <a class="fd-button" href="https://github.com/Tencent-Hunyuan/HY-WorldPlay" target="_blank" rel="noopener noreferrer">Official code</a>
</div>

Introduced by [Tencent Hunyuan](https://github.com/Tencent-Hunyuan/HY-WorldPlay), HY-WorldPlay is a
real-time interactive image-to-video (I2V) world model with action + camera-trajectory conditioning and
reconstituted-context memory. FlashDreams ships a native port of the distilled WAN-5B variant (Wan 2.2
TI2V-5B backbone, 4-step distilled Euler).

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-2.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<figcaption class="tiny-figcaption">
  Generated with FlashDreams' native HY-WorldPlay Implementation.
</figcaption>

## Quick Start



### Cam2V

```bash
uv sync --package flashdreams-hy-worldplay --inexact
uv run --no-sync flashdreams-run-v2 cam2v-hy-worldplay \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

- [Demo presets](#cam2v-presets)
- [Demo arguments](../demos/cam2v.md#demo-arguments)

<hr>

## Demo Presets

<a id="cam2v-presets"></a>

### Cam2V

| Preset | Description |
| --- | --- |
| `cam2v-hy-worldplay` | HY-WorldPlay WAN-5B I2V with live camera-history adaptation. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/hy_worldplay">Integration source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/hy_worldplay/config.py">Pipeline configurations</a>
</div>

- **GPU:** A CUDA-capable NVIDIA GPU

<hr>

## Performance (Outdated)

Here is the profiling benchmark on total DiT + VAE-decode runtime for FlashDreams HY-WorldPlay
compared to the [official HY-WorldPlay implementation](https://github.com/Tencent-Hunyuan/HY-WorldPlay)
under matched settings.

<figure class="benchmark-figure-wrap">
  <div
    id="hy-worldplay-benchmark-chart"
    class="benchmark-figure"
  data-benchmark-json-url="../../_static/performance/hy_worldplay/perf-0530.json"
  data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
    data-chart-aria-label="HY-WorldPlay benchmark chart"
  ></div>
</figure>
<figcaption class="tiny-figcaption">
  This chart shows total DiT + VAE-decode runtime per autoregressive chunk (4 diffusion steps) in
  milliseconds, at steady state (median of the post-warmup chunks), measured at num_chunk=8,
  704x1280, seed=0 on a single GB300. For an apples-to-apples comparison, both implementations are
  forced to use the cuDNN attention backend and torch.compile under matched runtime settings.
  For the official HY-WorldPlay implementation, see
  <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/hy_worldplay/tests/parity_check">this instruction</a>.
</figcaption>
<script src="../_static/js/benchmark_chart.js"></script>


<hr>

## Samples

<div class="model-video-grid">
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-4.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption class="tiny-figcaption">
      Walking through a seaside village
    </figcaption>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-8.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption class="tiny-figcaption">
      Walking through a snowy forest
    </figcaption>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-9.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption class="tiny-figcaption">
      Walking toward a castle
    </figcaption>
  </div>
</div>

<hr>

## Citation

If you use HY-WorldPlay, cite the original work:

```bibtex
@misc{sun2025worldplay,
  title         = {WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling},
  author        = {Sun, Wenqiang and Zhang, Haiyu and Wang, Haoyuan and Wu, Junta and Wang, Zehan and Wang, Zhenwei and Wang, Yunhong and Zhang, Jun and Wang, Tengfei and Guo, Chunchao},
  year          = {2025},
  eprint        = {2512.14614},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2512.14614},
  url           = {https://arxiv.org/abs/2512.14614},
}
```
