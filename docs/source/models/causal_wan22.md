---
title: 'Causal Wan2.2'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers">Model weights</a>
  <a class="fd-button" href="https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_t2v.py">Official code</a>
</div>

CausalWan2.2 is a [FastVideo](https://github.com/hao-ai-lab/FastVideo)-released
14B MoE causal-diffusion variant of Wan 2.2 with 8-step inference.

<div class="fd-card fd-model-card">
  <img class="fd-card-preview" src="../_static/model_clips/causal_wan22/fastvideo-causal-wan2.2-t2v-14b-1.avif" alt="" />
</div>

<figcaption class="tiny-figcaption">
  Teaser video generated with the FlashDreams Causal Wan2.2 integration.
</figcaption>

This integration uses `flashdreams-run-v2`.

## Quick Start



### T2V

```bash
uv sync --package flashdreams-fastvideo-causal-wan22 --inexact
uv run --no-sync flashdreams-run-v2 \
  t2v-fastvideo-causal-wan2.2-t2v-14b \
  --output-path artifacts/causal-wan22.mp4 --timeout unbound -- \
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
| `t2v-fastvideo-causal-wan2.2-t2v-14b` | FastVideo CausalWan 2.2 14B MoE T2V with the Wan VAE decoder and eight denoising steps. |

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/fastvideo_causal_wan22">Integration Source</a>
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/fastvideo_causal_wan22/config.py">Pipeline configurations</a>
</div>

- **Minimum VRAM:** about 112 GB.
- **PyTorch:** 2.9 or newer.

<hr>

## Samples

<div class="model-media-grid zoomable">
  <div class="fd-card fd-model-card">
    <img class="fd-card-preview" src="../_static/model_clips/causal_wan22/fastvideo-causal-wan2.2-t2v-14b-1.avif" alt="" />
    <figcaption class="tiny-figcaption">
      prompt: "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."
    </figcaption>
  </div>
  <div class="fd-card fd-model-card">
    <img class="fd-card-preview" src="../_static/model_clips/causal_wan22/fastvideo-causal-wan2.2-t2v-14b-2.avif" alt="" />
    <figcaption class="tiny-figcaption">
      prompt: "A playful raccoon is seen playing an electronic guitar, strumming the strings with its front paws. The raccoon has distinctive black facial markings and a bushy tail. It sits comfortably on a small stool, its body slightly tilted as it focuses intently on the instrument. The setting is a cozy, dimly lit room with vintage posters on the walls, adding a retro vibe. The raccoon's expressive eyes convey a sense of joy and concentration. Medium close-up shot, focusing on the raccoon's face and hands interacting with the guitar."
    </figcaption>
  </div>
</div>

<hr>

## Citation

FastVideo lists the following research citations:

```bibtex

@article{zhang2025fast,
  title={Fast video generation with sliding tile attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Su, Runlong and Ding, Hangliang and Stoica, Ion and Liu, Zhengzhong and Zhang, Hao},
  journal={arXiv preprint arXiv:2502.04507},
  year={2025}
}

@article{zhang2025vsa,
  title={Vsa: Faster video diffusion with trainable sparse attention},
  author={Zhang, Peiyuan and Chen, Yongqi and Huang, Haofeng and Lin, Will and Liu, Zhengzhong and Stoica, Ion and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2505.13389},
  year={2025}
}
```
