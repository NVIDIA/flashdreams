---
title: 'SANA-WM'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-cta-row">
  <a class="fd-button" href="https://nvlabs.github.io/Sana/">Project page</a>
  <a class="fd-button" href="https://arxiv.org/abs/2410.10629">arXiv paper</a>
  <a class="fd-button" href="https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming">Checkpoint</a>
  <a class="fd-button" href="https://github.com/NVlabs/Sana">Official code</a>
</div>

SANA-WM is NVlabs/Sana's camera-controlled world model family. FlashDreams
includes the chunk-causal streaming release, exposed through the
`cam2v-sana-wm-streaming` application, and the full-sequence bidirectional
release, exposed as a programmatic pipeline configuration.


<img alt="SANA-WM streaming FlashDreams sample clip." src="../_static/model_clips/sana_wm/sana-wm-streaming.avif" />
<figcaption class="tiny-figcaption">
  Generated via Flashdreams SANA-WM streaming demo using example assets via `--example-data` flag.
</figcaption>

<div class="transparent-section" markdown>
## Quick Start



### Cam2V

```bash
uv sync --package flashdreams-sana-wm --inexact
uv run --no-sync flashdreams-run-v2 cam2v-sana-wm-streaming \
  --mode webrtc --host 0.0.0.0 --port 8089 -- --example-data
```

- [Demo presets](#cam2v-presets)
- [Demo arguments](../demos/cam2v.md#demo-arguments)

</div>

<div class="transparent-section" markdown>
## Demo Presets

<a id="cam2v-presets"></a>

### Cam2V

| Preset | Description |
| --- | --- |
| `cam2v-sana-wm-streaming` | Chunk-causal SANA-WM at 1280 x 704 with ten 24-frame blocks by default. |

</div>

<hr>

## Developer Details

<div class="fd-cta-row">
  <a class="fd-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations_v2/sana_wm">Integration source</a>
</div>

- **PyTorch:** 2.9 or newer.
- **Precision:** BF16 by default. FP8 Stage-1/refiner inference is available on
  Hopper or newer GPUs (`sm_90+`), and FP4 is available on Blackwell
  (`sm_100+`). These upstream precision flags belong to
  `SANA-WM_streaming`.

## Performance (Outdated)

The charts below compare steady-state generation latency per produced chunk for
FlashDreams `SANA-WM_streaming` and the official `SANA-WM_streaming`
implementation under matched settings. Warmup runs and the first decoded chunk
are excluded from the headline metric. These GB300 latency runs show the
official implementation faster than FlashDreams for BF16, FP8, and FP4.

In these charts, `Official Impl` means the pinned NVlabs/Sana upstream
implementation measured by the FlashDreams benchmark harness under matched
settings. It is not the SANA-WM 80-scene benchmark result published by the
model authors.

 <figure class="benchmark-figure-wrap">
   <div
     id="sana-wm-streaming-bf16-benchmark-chart"
     class="benchmark-figure"
     data-benchmark-json-url="../../_static/performance/sana_wm_streaming/perf-0801-bf16.json"
     data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="SANA-WM streaming BF16 benchmark chart"
   ></div>
   <figcaption class="tiny-figcaption">
      BF16 steady-state milliseconds per produced chunk on one NVIDIA GB300 GPU:
      official 1,170.29 ms, FlashDreams 1,957.93 ms.
   </figcaption>
 </figure>

 <figure class="benchmark-figure-wrap">
   <div
     id="sana-wm-streaming-fp8-benchmark-chart"
     class="benchmark-figure"
     data-benchmark-json-url="../../_static/performance/sana_wm_streaming/perf-0801-fp8.json"
     data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="SANA-WM streaming FP8 benchmark chart"
   ></div>
   <figcaption class="tiny-figcaption">
      FP8 steady-state milliseconds per produced chunk on one NVIDIA GB300 GPU:
      official 1,482.92 ms, FlashDreams 2,392.67 ms.
   </figcaption>
 </figure>

 <figure class="benchmark-figure-wrap">
   <div
     id="sana-wm-streaming-fp4-benchmark-chart"
     class="benchmark-figure"
     data-benchmark-json-url="../../_static/performance/sana_wm_streaming/perf-0801-fp4.json"
     data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="SANA-WM streaming FP4 benchmark chart"
   ></div>
   <figcaption class="tiny-figcaption">
      FP4 steady-state milliseconds per produced chunk on one NVIDIA GB300 GPU:
      official 1,594.33 ms, FlashDreams 4,118.36 ms.
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>

All charts use the same demo image/prompt, `w-80,dw-40,w-80,aw-40`
action path, 241 requested frames, one discarded warmup run, and three measured
runs. The benchmark runs recorded FlashDreams commit bd0816e and upstream
commit 6298508.

### Bidirectional variant

`SANA-WM_bidirectional` renders a complete clip in one pass from a first frame,
prompt, and camera trajectory. It has no registered V2 application slug; use
the lower-level pipeline configuration:

The bidirectional model is available as a pipeline configuration:

```python

from sana_wm.config import PIPELINE_SANA_WM_BIDIRECTIONAL

pipeline = PIPELINE_SANA_WM_BIDIRECTIONAL.setup().to("cuda").eval()

```

This is the integration's lower-level pipeline configuration.

#### Bidirectional profiling benchmark

The BF16 chart below compares steady-state in-process generation latency per
generated clip for FlashDreams `SANA-WM_bidirectional` and the official
`SANA-WM_bidirectional` implementation under matched settings on one NVIDIA
GB300 GPU. FlashDreams measured 34,182.39 ms per clip versus 56,932.83 ms for
the official implementation.

In this chart, `Official Impl` means the pinned NVlabs/Sana upstream
implementation measured by the FlashDreams benchmark harness under matched
settings. It is not the SANA-WM 80-scene benchmark result published by the
model authors.

 <figure class="benchmark-figure-wrap">
   <div
     id="sana-wm-bidirectional-bf16-benchmark-chart"
     class="benchmark-figure"
     data-benchmark-json-url="../../_static/performance/sana_wm_bidirectional/perf-0801-bf16.json"
     data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
     data-chart-aria-label="SANA-WM bidirectional BF16 benchmark chart"
   ></div>
   <figcaption class="tiny-figcaption">
       This chart shows steady-state in-process generation latency per generated clip in milliseconds for a
       121-frame full-pipeline BF16 run (Stage-1 DiT + LTX-2 refiner + SANA VAE decode).
       The measured row used one NVIDIA GB300 GPU, one live warmup generation,
       and three measured generations.
       Model construction, checkpoint loading, video writing, and frame dumps are outside the timing boundary.
       The benchmark runs recorded FlashDreams commit bd0816e and upstream commit 6298508.
   </figcaption>
 </figure>
<script src="../_static/js/benchmark_chart.js"></script>

## Samples

<img alt="SANA-WM bidirectional FlashDreams sample clip." src="../_static/model_clips/sana_wm/sana-wm-bidirectional.avif" />

## Citation

If you use SANA-WM, please cite the original SANA work:

```bibtex

@misc{xie2024sana,
      title={SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers},
      author={Enze Xie and Junsong Chen and Junyu Chen and Han Cai and Haotian Tang and Yujun Lin and Zhekai Zhang and Muyang Li and Ligeng Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2410.10629},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
