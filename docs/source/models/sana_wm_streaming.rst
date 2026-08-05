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
FlashDreams runs it through the ``sana-wm-streaming`` runner.

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

Running the method
------------------

Launch the ``sana-wm-streaming`` runner with a first-frame image, a prompt, and
a camera trajectory:

.. code-block:: bash

   uv run flashdreams-run sana-wm-streaming \
       --image-path <path to initial frame PNG> \
       --prompt-path <path to prompt TXT> \
       --camera-path <path to camera trajectory NPY> \
       --intrinsics-path <path to intrinsics NPY> \
       --num-frames 241 \
       --output-dir outputs/sana_wm_streaming_bf16

The first frame, prompt, camera, and intrinsics inputs must follow the same
shape conventions as the ``SANA-WM_streaming`` release examples.

The runner defaults to 3 latent frames per block and the distilled schedule
``1000,960,889,727,0``. Requested frame counts are snapped to
``8 * --num-frame-per-block * k + 1`` before inference.

Optional inputs and knobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``--intrinsics-path`` is optional. When omitted, intrinsics are derived from
  the first-frame size.
- ``--camera-path`` can be replaced by an ``--action`` DSL string:

  .. code-block:: bash

     uv run flashdreams-run sana-wm-streaming \
         --image-path my_frame.png \
         --prompt "a scene description; describe the world's own motion" \
         --action "w-80,dw-40,w-80,aw-40" \
         --num-frames 241 \
         --output-dir outputs/mine_streaming

  Action trajectories are fitted to the snapped frame count: shorter action
  strings repeat, and longer action strings are truncated without materializing
  frames beyond the requested output length.

- ``--stage1-precision`` and ``--refiner-precision`` accept ``bf16``, ``fp8``,
  or ``fp4`` when the selected hardware supports the requested precision.

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run flashdreams-run sana-wm-streaming --help

What to expect
--------------

- **Model checkpoint**: pulled from
  ``huggingface.co/Efficient-Large-Model/SANA-WM_streaming`` on first run.
- **First launch**: a few minutes for download and warmup; subsequent launches
  reuse local caches.
- **Outputs**: ``outputs/<output-dir>/sana-wm-streaming.mp4``.

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
