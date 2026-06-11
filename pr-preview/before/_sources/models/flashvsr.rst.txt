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

FlashVSR
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://zhuang2002.github.io/FlashVSR/" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://arxiv.org/abs/2510.12747" target="_blank" rel="noopener noreferrer">arXiv paper</a>
     <a class="model-link-button" href="https://github.com/OpenImagingLab/FlashVSR" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

FlashVSR is a one-diffusion-step streaming diffusion framework for real-time video
super-resolution (VSR). It combines a train-friendly three-stage distillation pipeline,
locality-constrained sparse attention that bridges the train-test resolution
gap, and a tiny conditional decoder for fast reconstruction.

.. image:: https://github.com/OpenImagingLab/FlashVSR/raw/main/examples/WanVSR/assets/teaser.png
   :alt: FlashVSR teaser figure.
   :width: 100%

.. raw:: html

   <p class="model-footnote">
     Teaser image source:
     <a href="https://github.com/OpenImagingLab/FlashVSR">FlashVSR official repository</a>.
   </p>

Requirements
------------

- **Minimum VRAM**: ~24 GB.
- **PyTorch**: >= 2.9.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/flashvsr

Running the method
------------------

To run FlashVSR, provide an input video path and launch one of the registered
runner slugs via ``flashdreams-run``. For example:

.. code-block:: bash

   uv run --project integrations/flashvsr \
       flashdreams-run \
       flashvsr-v1.1-sparse-ratio-2.0 \
       --input-path https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/main/examples/WanVSR/inputs/example1.mp4 \
       --chunk-size 8

For multi-GPU inference, use the dense full-attention preset with ``torchrun``
on top of ``uv run flashdreams-run`` (taking 4 GPUs as an example):

.. code-block:: bash

   uv run --project integrations/flashvsr \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       flashvsr-v1.1-full-attn \
       --input-path https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/main/examples/WanVSR/inputs/example1.mp4 \
       --chunk-size 8

.. note::

   Multi-GPU is supported only by the dense ``flashvsr-v1.1-full-attn`` preset.
   The ``flashvsr-v1.1-sparse-ratio-*`` presets are single-GPU only because
   their Triton sparse-attention backend is not context-parallel aware.

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``flashvsr-v1.1-sparse-ratio-2.0``
     - Streaming 2x video super-resolution with the stable sparse-attention preset.
   * - ``flashvsr-v1.1-sparse-ratio-1.5``
     - Streaming 2x video super-resolution with the faster sparse-attention preset.
   * - ``flashvsr-v1.1-full-attn``
     - Dense full-attention preset with multi-GPU context-parallel support.

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/flashvsr \
       flashdreams-run \
       flashvsr-v1.1-sparse-ratio-2.0 \
       --help

A generated sample from the above commands:

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
     <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
       <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/flashvsr/flashvsr-v1.1-sparse-ratio-2.0.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
     <video autoplay muted loop playsinline preload="metadata" style="position: absolute; left: 10px; bottom: 10px; width: 50%; border: 2px solid rgba(255, 255, 255, 0.9); border-radius: 8px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5); pointer-events: none;">
       <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/flashvsr/example1.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
     <div class="model-video-overlay">
       FlashVSR 2x output (1280x768) from <code>flashvsr-v1.1-sparse-ratio-2.0</code>;
       low-resolution input (672x384) inset at bottom-left.
       Input from the
       <a href="https://github.com/OpenImagingLab/FlashVSR/tree/main/examples/WanVSR/inputs">FlashVSR examples</a>.
     </div>
   </div>

Profiling benchmark
-------------------

Here is the profiling benchmark on per-chunk 2x upsampling time for FlashDreams FlashVSR
compared to the `official FlashVSR implementation <https://github.com/OpenImagingLab/FlashVSR>`_
under matched settings.

.. raw:: html

  <figure class="benchmark-figure-wrap">
    <div
      id="flashvsr-benchmark-chart"
      class="benchmark-figure"
      data-benchmark-md-url="../_static/performance/flashvsr/perf-0527.md"
      data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
      data-chart-aria-label="FlashVSR benchmark chart"
    ></div>
    <figcaption>
      <p class="model-footnote">
        This chart shows per-chunk 2x upsampling time in milliseconds on a single GB200 GPU with a chunk size of 8 frames.
        For the official FlashVSR implementation, see
        <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/flashvsr/tests/parity_check">this instruction</a>.
      </p>
    </figcaption>
  </figure>
  <script src="../_static/js/benchmark_chart.js"></script>

Citation
--------

If you use FlashVSR, please cite the original work:

.. code-block:: bibtex

   @inproceedings{zhuang2026flashvsr,
     title={FlashVSR: Towards Real-time Diffusion-Based Streaming Video Super Resolution},
     author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
     booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
     pages={43482--43493},
     year={2026}
   }
