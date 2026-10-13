---
title: 'Self-Forcing'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://self-forcing.github.io/">Project page</a>
  <a class="fd-button" href="https://arxiv.org/abs/2506.08009">arXiv paper</a>
  <a class="fd-button" href="https://github.com/guandeh17/Self-Forcing">Official code</a>
</div>

Self-Forcing is a text-to-video (T2V) model based on [Wan2.1](wan21.md).
It uses a training paradigm for autoregressive video diffusion that simulates
inference-time rollout during training with KV caching, reducing the train-test
gap and enabling efficient streaming generation quality.

![Self-Forcing teaser figure.](https://self-forcing.github.io/static/teaser.jpg)

<figcaption class="tiny-figcaption">
  Teaser image source:
  <a href="https://self-forcing.github.io/">Self-Forcing project page</a>.
</figcaption>

## Quick Start



### T2V

```bash
uv sync --package flashdreams-self-forcing --inexact
uv run --no-sync flashdreams-run-v2 \
  t2v-self-forcing-wan2.1-t2v-1.3b \
  --output-path artifacts/self-forcing.mp4 --timeout unbound -- \
  --prompt "A cat surfing" --total-blocks 7
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
| `t2v-self-forcing-wan2.1-t2v-1.3b` | Official checkpoint and Wan VAE decoder. |
| `t2v-self-forcing-wan2.1-t2v-1.3b-taehv` | Official checkpoint with the faster TAEHV decoder. |
| `t2v-self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope` | Long-rollout preset with static sink 5, rolling window 7, and KV-cache-relative RoPE. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/self_forcing">Integration Source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/self_forcing/config.py">Pipeline configurations</a>
</div>

- **Minimum VRAM:** about 24 GB.
- **PyTorch:** 2.9 or newer.

<hr>

## Performance (Outdated)

Here is the profiling benchmark on total DiT runtime for FlashDreams Self-Forcing compared to
the [official Self-Forcing implementation](https://github.com/guandeh17/Self-Forcing)
and the [FastVideo implementation](https://github.com/hao-ai-lab/FastVideo)
under matched settings.

<figure class="benchmark-figure-wrap">
  <div
    id="self-forcing-benchmark-chart"
    class="benchmark-figure"
  data-benchmark-json-url="../../_static/performance/self_forcing/perf-0521.json"
    data-benchmark-series="fastvideo:FastVideo:#f59e0b;official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
    data-chart-aria-label="Self-Forcing benchmark chart"
  ></div>
  <figcaption class="tiny-figcaption">
    This chart shows the DiT total runtime (4 denoising steps in milliseconds) at the 6th autoregressive rollout on a single GPU.
    For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend and <code>torch.compile</code> for DiT network.
    For profiling the official implementation, see
    <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/self_forcing/tests/parity_check/README.md">this instruction</a>.
    For profiling the FastVideo implementation, see
    <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/self_forcing/tests/baseline_fastvideo/README.md">this instruction</a>.
  </figcaption>
</figure>
<script src="../_static/js/benchmark_chart.js"></script>

<hr>

## Citation

If you use Self-Forcing, please cite the original work:

```bibtex

@article{huang2025self,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={167283--167308},
  year={2025}
}
```
