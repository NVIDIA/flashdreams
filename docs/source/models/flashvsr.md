---
title: 'FlashVSR'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://zhuang2002.github.io/FlashVSR/">Project page</a>
  <a class="fd-button" href="https://arxiv.org/abs/2510.12747">arXiv paper</a>
  <a class="fd-button" href="https://github.com/OpenImagingLab/FlashVSR">Official code</a>
</div>

FlashVSR is a one-diffusion-step streaming diffusion framework for real-time video
super-resolution (VSR). It combines a train-friendly three-stage distillation pipeline,
locality-constrained sparse attention that bridges the train-test resolution
gap, and a tiny conditional decoder for fast reconstruction.

![FlashVSR teaser figure.](https://github.com/OpenImagingLab/FlashVSR/raw/main/examples/WanVSR/assets/teaser.png)

<figcaption class="tiny-figcaption">
  Teaser image source:
  <a href="https://github.com/OpenImagingLab/FlashVSR">FlashVSR official repository</a>.
</figcaption>

## Quick Start



### V2V

```bash
uv sync --package flashdreams-flashvsr --inexact
uv run --no-sync flashdreams-run-v2 \
  v2v-flashvsr-v1.1-sparse-ratio-2.0 \
  --output-path artifacts/flashvsr.mp4 --timeout unbound -- --video-path input.mp4
```

- [Demo presets](#v2v-presets)
- [Demo arguments](../demos/v2v.md#demo-arguments)

<hr>

## Demo Presets

<a id="v2v-presets"></a>

### V2V

| Preset | Description |
| --- | --- |
| `v2v-flashvsr-v1.1-sparse-ratio-2.0` | Stable sparse-attention 2x video super-resolution. |
| `v2v-flashvsr-v1.1-sparse-ratio-1.5` | Faster sparse-attention 2x video super-resolution. |
| `v2v-flashvsr-v1.1-full-attn` | Dense full attention with multi-GPU context-parallel support. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/flashvsr">Integration source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/flashvsr/config.py">Pipeline configurations</a>
</div>

- **Minimum VRAM:** about 24 GB.
- **Python:** 3.10 or newer.
- **PyTorch:** 2.9 or newer.

### Post-processing presets

Cam2V and Interactive Drive commands select these presets with
`--postprocess-preset PRESET` in their application arguments, after `--`:

| Preset | Description |
| --- | --- |
| `flashvsr-v1.1-sparse-2.0` | N/A |
| `flashvsr-v1.1-sparse-1.5` | N/A |
| `flashvsr-v1.1-full-attn` | N/A |

<hr>

## Performance (Outdated)

This historical benchmark compares per-chunk 2x upsampling time for FlashDreams
FlashVSR with the [official FlashVSR implementation](https://github.com/OpenImagingLab/FlashVSR)
under matched settings.

<figure class="benchmark-figure-wrap">
  <div
    id="flashvsr-benchmark-chart"
    class="benchmark-figure"
    data-benchmark-json-url="../../_static/performance/flashvsr/perf-0527.json"
    data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
    data-chart-aria-label="FlashVSR benchmark chart"
  ></div>
  <figcaption class="tiny-figcaption">
      This chart records per-chunk 2x upsampling time in milliseconds on a single GB200 GPU with a chunk size of 8 frames.
      The current v2 application presets use 16-frame steady-state chunks, so these historical measurements are not timings of the current defaults.
      For the official FlashVSR implementation, see
      <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/flashvsr/tests/parity_check">this instruction</a>.
    </figcaption>
</figure>
<script src="../_static/js/benchmark_chart.js"></script>

<hr>

## Samples

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/flashvsr/flashvsr-v1.1-sparse-ratio-2.0.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <figcaption class="tiny-figcaption">
    FlashVSR 2x output (1280x768) from <code>flashvsr-v1.1-sparse-ratio-2.0</code>;
    low-resolution input (672x384) inset at bottom-left.
    Input from the
    <a href="https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/inputs">FlashVSR examples</a>.
  </figcaption>
</div>

<hr>

## Citation

If you use FlashVSR, please cite the original work:

```bibtex

@inproceedings{zhuang2026flashvsr,
  title={FlashVSR: Towards Real-time Diffusion-Based Streaming Video Super Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={43482--43493},
  year={2026}
}
```
