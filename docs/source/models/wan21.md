---
title: 'Wan2.1'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://wan.video/">Project page</a>
  <a class="fd-button" href="https://arxiv.org/abs/2503.20314">arXiv paper</a>
  <a class="fd-button" href="https://github.com/Wan-Video/Wan2.1">Official code</a>
</div>

Wan2.1 is a bidirectional video generation model, supporting both
text-to-video (T2V) and image-to-video (I2V) tasks.

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/model_clips/wan21/wan21-t2v-1.3b-480p.avif" alt="" />
  <figcaption class="tiny-figcaption">
    Generated via FlashDreams with the prompt: "two cats dancing together in a circle in the rain, in a rainforest"
  </figcaption>
</div>

## Quick Start


### T2V

```bash
uv sync --package flashdreams-wan21 --inexact
uv run --no-sync flashdreams-run-v2 \
  t2v-wan21-t2v-1.3b-480p \
  --output-path artifacts/wan21.mp4 --timeout unbound -- --prompt "A cat surfing"
```

<div class="fd-cta-row">
  <a class="fd-button" href="#t2v-presets">Demo Presets</a>
  <a class="fd-button" href="../demos/t2v.md#demo-arguments">Demo Arguments</a>
</div>

<hr>

## Demo Presets

<a id="t2v-presets"></a>

### T2V

| Preset | Description |
| --- | --- |
| `t2v-wan21-t2v-1.3b-480p` | Wan 2.1 T2V 1.3B at 480p with a single bidirectional block. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/wan21">Integration Source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/wan21/config.py">Pipeline configurations</a>
</div>

- **Minimum VRAM:** about 46 GB.
- **PyTorch:** 2.9 or newer.

### Pipeline configurations

| Method | Description |
| --- | --- |
| `wan21-t2v-1.3b-480p` | Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only). |
| `wan21-i2v-14b-480p` | Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame). |

<hr>

## Performance (Outdated)

Here is the profiling benchmark on DiT per-step runtime for FlashDreams Wan2.1
compared to the [official Wan2.1 implementation](https://github.com/Wan-Video/Wan2.1)
and the [FastVideo](https://github.com/hao-ai-lab/FastVideo) baseline under
matched settings.

 <figure class="benchmark-figure-wrap">
   <div
     id="wan21-benchmark-chart"
     class="benchmark-figure"
    data-benchmark-json-url="../../_static/performance/wan21/perf-0521.json"
     data-benchmark-series="fastvideo:FastVideo:#f59e0b;official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="Wan2.1 benchmark chart"
   ></div>
   <figcaption>
      This chart shows per-diffusion-step DiT runtime in milliseconds with CFG at 480p (81 frames) on a single GPU.
      For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend under matched runtime settings.
      For the official Wan2.1 implementation, see
      <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/wan21/tests/parity_check">this instruction</a>.
      For the FastVideo baseline, see
      <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/wan21/tests/baseline_fastvideo">this instruction</a>.
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>

<hr>

## Samples

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/model_clips/wan21/wan21-i2v-14b-480p.avif" alt="" />
  <figcaption class="tiny-figcaption">
    prompt: "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    <br/>
    image: https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/examples/i2v_input.JPG
  </figcaption>
</div>

<hr>

## Citation

If you use Wan2.1, please cite the original work:

```bibtex

@article{wan2025,
      title={Wan: Open and Advanced Large-Scale Video Generative Models},
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      journal={arXiv preprint arXiv:2503.20314},
      year={2025}
}
```
