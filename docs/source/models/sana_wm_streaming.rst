.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

SANA-WM_streaming
===================================

.. container:: fd-cta-row

   .. button-link:: https://nvlabs.github.io/Sana/
      :color: primary

      Project page

   .. button-link:: https://arxiv.org/abs/2410.10629
      :color: primary

      arXiv paper

   .. button-link:: https://huggingface.co/Efficient-Large-Model/SANA-WM_streaming
      :color: primary

      Checkpoint

   .. button-link:: https://github.com/NVlabs/Sana
      :color: primary

      Official code

``SANA-WM_streaming`` is the chunk-causal, camera-controlled
`NVlabs/Sana <https://github.com/NVlabs/Sana>`_ world model release. It
produces video progressively across autoregressive chunks with a chunk-causal
Stage-1 DiT, streaming LTX-2 refiner, and streaming VAE decode path.
FlashDreams exposes it through the ``cam2v-sana-wm-streaming`` application.

The sibling full-sequence release has a separate model card:
:doc:`sana_wm_bidirectional`.

.. container:: model-video-card model-hero-media zoomable

   .. image:: /_static/model_clips/sana_wm/sana-wm-streaming.avif
      :alt: SANA-WM streaming FlashDreams sample clip.
      :class: model-video-player

Requirements
------------

- **PyTorch**: >= 2.9.
- **Precision**: BF16 by default. FP8 Stage-1/refiner inference is available on
  Hopper or newer GPUs (``sm_90+``), and FP4 is available on Blackwell
  (``sm_100+``). These upstream precision flags belong to
  ``SANA-WM_streaming``.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --package flashdreams-sana-wm --extra dev

Interactive Cam2V application
-----------------------------

Launch the V2 application to drive SANA-WM with live keyboard controls. The
model adapter passes controls through the SANA-WM action remapper and appends
each generated block to the camera conditioning history.

.. code-block:: bash

   uv run --no-sync flashdreams-run-v2 cam2v-sana-wm-streaming \
       --mode webrtc --host 0.0.0.0 --port 8089 -- \
       --example-data

The application uses the checkpoint fixed resolution of 1280x704 and ten
24-frame blocks by default. Use ``--total-blocks`` after ``--`` to change the
rollout length.

Use ``--example-data`` to download the official ``demo_0.png`` and paired prompt
to the FlashDreams example-data cache. Explicit image and prompt arguments
override those example inputs.

Profiling benchmark
-------------------

The charts below compare steady-state generation latency per produced chunk for
FlashDreams ``SANA-WM_streaming`` and the official ``SANA-WM_streaming``
implementation under matched settings. Warmup runs and the first decoded chunk
are excluded from the headline metric. These GB300 latency runs show the
official implementation faster than FlashDreams for BF16, FP8, and FP4.

In these charts, ``Official Impl`` means the pinned NVlabs/Sana upstream
implementation measured by the FlashDreams benchmark harness under matched
settings. It is not the SANA-WM 80-scene benchmark result published by the
model authors.

.. raw:: html

   <figure class="benchmark-figure-wrap">
     <div
       id="sana-wm-streaming-bf16-benchmark-chart"
       class="benchmark-figure"
       data-benchmark-md-url="../_static/performance/sana_wm_streaming/perf-0801-bf16.md"
       data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="SANA-WM streaming BF16 benchmark chart"
     ></div>
     <figcaption>
       <p class="model-footnote">
         BF16 steady-state milliseconds per produced chunk on one NVIDIA GB300 GPU:
         official 1,170.29 ms, FlashDreams 1,957.93 ms.
       </p>
     </figcaption>
   </figure>

   <figure class="benchmark-figure-wrap">
     <div
       id="sana-wm-streaming-fp8-benchmark-chart"
       class="benchmark-figure"
       data-benchmark-md-url="../_static/performance/sana_wm_streaming/perf-0801-fp8.md"
       data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="SANA-WM streaming FP8 benchmark chart"
     ></div>
     <figcaption>
       <p class="model-footnote">
         FP8 steady-state milliseconds per produced chunk on one NVIDIA GB300 GPU:
         official 1,482.92 ms, FlashDreams 2,392.67 ms.
       </p>
     </figcaption>
   </figure>

   <figure class="benchmark-figure-wrap">
     <div
       id="sana-wm-streaming-fp4-benchmark-chart"
       class="benchmark-figure"
       data-benchmark-md-url="../_static/performance/sana_wm_streaming/perf-0801-fp4.md"
       data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="SANA-WM streaming FP4 benchmark chart"
     ></div>
     <figcaption>
       <p class="model-footnote">
         FP4 steady-state milliseconds per produced chunk on one NVIDIA GB300 GPU:
         official 1,594.33 ms, FlashDreams 4,118.36 ms.
       </p>
     </figcaption>
   </figure>
  <script src="../_static/js/benchmark_chart.js"></script>

All charts use the same demo image/prompt, ``w-80,dw-40,w-80,aw-40``
action path, 241 requested frames, one discarded warmup run, and three measured
runs. The benchmark runs recorded FlashDreams commit bd0816e and upstream
commit 6298508.

Citation
--------

If you use SANA-WM, please cite the original SANA work:

.. code-block:: bibtex

   @misc{xie2024sana,
         title={SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers},
         author={Enze Xie and Junsong Chen and Junyu Chen and Han Cai and Haotian Tang and Yujun Lin and Zhekai Zhang and Muyang Li and Ligeng Zhu and Yao Lu and Song Han},
         year={2024},
         eprint={2410.10629},
         archivePrefix={arXiv},
         primaryClass={cs.CV}
   }
