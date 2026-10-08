---
title: 'LingBot-World'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://technology.robbyant.com/lingbot-world">Project page</a>
  <a class="fd-button" href="https://github.com/robbyant/lingbot-world">Official code</a>
</div>

Introduced by [Robbyant](https://technology.robbyant.com/), LingBot-World is a camera-controllable image-to-video
(I2V) world model with streaming inference and context-parallel runtime support. This page covers the original
[LingBot-World v1](https://github.com/robbyant/lingbot-world) and the causal-fast
[LingBot-World v2](https://github.com/Robbyant/lingbot-world-v2) checkpoints in 14B and 1.3B sizes.

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="https://gw.alipayobjects.com/v/huamei_u94ywh/afts/video/XQk7Rb44qJwAAAAAgfAAAAgAfoeUAQBr" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>
<figcaption class="tiny-figcaption">
  Teaser video source:
  <a href="https://technology.robbyant.com/lingbot-world">LingBot-World project page</a>.
</figcaption>

## Quick Start



### Cam2V

```bash
uv sync --package flashdreams-lingbot --inexact
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data

# `--example-data` downloads `image.jpg`,
# `intrinsics.npy`, `poses.npy`, and `prompt.txt` from the
# canonical examples folder: https://github.com/Robbyant/lingbot-world-v2/tree/main/examples
```

- [Demo presets](#cam2v-presets)
- [Demo arguments](../demos/cam2v.md#demo-arguments)

<hr>

## Demo Presets

<a id="cam2v-presets"></a>

### Cam2V

| Preset | Description |
| --- | --- |
| `cam2v-lingbot` | Compatibility alias for the bounded-window TAEHV default. |
| `cam2v-lingbot-world-fast` | LingBot-World v1 with the Wan VAE decoder and full KV cache. |
| `cam2v-lingbot-world-fast-taehv-window15-sink3` | LingBot-World v1 with TAEHV and bounded streaming KV cache. |
| `cam2v-lingbot-world-v2-14b-causal-fast` | LingBot-World v2 14B causal-fast with the Wan VAE decoder. |
| `cam2v-lingbot-world-v2-14b-causal-fast-taehv-window15-sink3` | LingBot-World v2 14B causal-fast with TAEHV and bounded streaming KV cache. |
| `cam2v-lingbot-world-v2-1p3b-causal-fast-perf` | LingBot-World v2 1.3B tuned for RTX 5090-class GPUs with row-wise FP8 and in-tree Sage attention. |
| `cam2v-lingbot-world-v2-1p3b-causal-fast-perf-taehv` | The 1.3B performance preset with TAEHV decoding. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/lingbot">Integration source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/lingbot/config.py">Pipeline configuration</a>
</div>

- **Minimum VRAM:** about 120 GB for the 14B presets; the 1.3B performance presets target 32 GB RTX 5090-class GPUs.
- **PyTorch:** 2.9 or newer.
- **Disk:** keep ~200 GB free for the 14B model and Hugging Face cache.
- **First launch:** a few minutes (download, compilation, Triton autotuning, and
  CUDA-graph warmup). Subsequent launches reuse the caches.

The 1.3B performance presets use row-wise FP8 linear layers and the in-tree
`sage` attention backend; they do not require the external SageAttention
package. The TAEHV variant replaces the Wan VAE decoder. These paths were
validated on Linux with an RTX 5090; exclude compilation warmup from
steady-state benchmarks.

<hr>

## Performance (Outdated)

Here is the profiling benchmark on total DiT runtime for FlashDreams LingBot-World
compared to the [official LingBot-World implementation](https://github.com/robbyant/lingbot-world)
and [LightX2V](https://github.com/ModelTC/lightx2v) under
matched settings.

 <figure class="benchmark-figure-wrap">
   <div
     id="lingbot-world-benchmark-chart"
     class="benchmark-figure"
    data-benchmark-json-url="../../_static/performance/lingbot_world/perf-0521.json"
    data-benchmark-series="official:Official Impl:#3b82f6;lightx2v:LightX2V:#f59e0b;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="LingBot-World benchmark chart"
   ></div>
   <figcaption class="tiny-figcaption">
      This chart shows total DiT runtime (4 diffusion steps) in milliseconds at the 6th autoregressive rollout on 4x GPUs.
      For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend under matched runtime settings,
      and all runs use Ulysses sequence parallelism for multi-GPU inference.
      The historical comparison used the upstream official implementation
      and LightX2V as the two external baselines.
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>


<hr>

## Samples

<div class="model-video-grid zoomable">
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-01.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption class="tiny-figcaption">
      example_idx: 01
    </figcaption>
  </div>
  <div class="model-video-card">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-02.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <figcaption class="tiny-figcaption">
      example_idx: 02
    </figcaption>
  </div>
</div>

<hr>

## Citation

If you use LingBot-World, please cite the original work:

```bibtex

@article{lingbot-world,
      title={Advancing Open-source World Models},
      author={Robbyant Team and Zelin Gao and Qiuyu Wang and Yanhong Zeng and Jiapeng Zhu and Ka Leong Cheng and Yixuan Li and Hanlin Wang and Yinghao Xu and Shuailei Ma and Yihang Chen and Jie Liu and Yansong Cheng and Yao Yao and Jiayi Zhu and Yihao Meng and Kecheng Zheng and Qingyan Bai and Jingye Chen and Zehong Shen and Yue Yu and Xing Zhu and Yujun Shen and Hao Ouyang},
      journal={arXiv preprint arXiv:2601.20540},
      year={2026}
}
```
