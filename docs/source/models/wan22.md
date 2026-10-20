---
title: 'Wan 2.2 TI2V-5B'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://wan.video/">Project page</a>
  <a class="fd-button" href="https://arxiv.org/abs/2503.20314">arXiv paper</a>
  <a class="fd-button" href="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers">Checkpoint</a>
  <a class="fd-button" href="https://github.com/Wan-Video/Wan2.2">Official code</a>
</div>

Wan 2.2 TI2V-5B is a bidirectional text-and-image-to-video model. Given a
prompt and first-frame image, it generates a complete 81-frame, 1280x640 clip
in one rollout. FlashDreams exposes it through the
`t2v-wan22-ti2v-5b` application.

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/model_clips/wan22/wan22-ti2v-5b.avif" alt="" />
</div>

<figcaption class="tiny-figcaption">
  FlashDreams output for the prompt <em>A cat surfing</em>, conditioned on the
  bundled FastVideo Causal Wan 2.2 first frame.
</figcaption>

## Quick Start



### T2V

```bash
uv sync --package flashdreams-wan22 --inexact
uv run --no-sync flashdreams-run-v2 \
  t2v-wan22-ti2v-5b --mode mp4 \
  --output-path artifacts/wan22.mp4 --timeout unbound -- \
  --prompt "A cat surfing" \
  --image-path integrations_v2/wan22/apps/t2v/assets/fastvideo-causal-wan22-cat-surfing-first-frame.png \
  --no-ui --no-compile
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
| `t2v-wan22-ti2v-5b` | Wan 2.2 TI2V-5B at 1280 x 640 with a required first frame and one generated block. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/wan22">Integration Source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/wan22/config.py">Pipeline configurations</a>
</div>

- **GPU:** CUDA-capable NVIDIA GPU.
- **PyTorch:** 2.9 or newer.

<hr>

## Citation

If you use Wan 2.2, cite the original work:

```bibtex
@article{wan2025,
  title   = {Wan: Open and Advanced Large-Scale Video Generative Models},
  author  = {Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
  journal = {arXiv preprint arXiv:2503.20314},
  year    = {2025},
  url     = {https://arxiv.org/abs/2503.20314},
}
```
