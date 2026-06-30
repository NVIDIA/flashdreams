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

HY-WorldPlay
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/Tencent-Hunyuan/HY-WorldPlay" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://github.com/Tencent-Hunyuan/HY-WorldPlay" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Introduced by `Tencent Hunyuan <https://github.com/Tencent-Hunyuan/HY-WorldPlay>`_, HY-WorldPlay is a
real-time interactive image-to-video (I2V) world model with action + camera-trajectory conditioning and
reconstituted-context memory. FlashDreams ships a native port of the distilled WAN-5B variant (Wan 2.2
TI2V-5B backbone, 4-step distilled Euler).

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
     <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
       <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-1.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>
   <p class="model-footnote">
     Generated with FlashDreams' native HY-WorldPlay WAN-5B I2V pipeline.
   </p>

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/hy_worldplay

Running the method
------------------

HY-WorldPlay WAN-5B is image-to-video only. Launch it via ``flashdreams-run``, passing a first-frame
image (or ``--example-data``) and the distilled checkpoint:

.. code-block:: bash

   uv run --project integrations/hy_worldplay \
       flashdreams-run \
       hy-worldplay-wan-i2v-5b \
       --example-data \
       --ckpt-path /path/to/models/wan_distilled_model/model.pt \
       --num-chunk 8 \
       --pose "w-31"

``--ckpt-path`` is the distilled ``model.pt`` from the gated
`tencent/HY-WorldPlay <https://huggingface.co/tencent/HY-WorldPlay>`_ repo (set ``HF_TOKEN`` first); omit it
to run the zero-init base Wan 2.2 identity. ``--pose`` is a camera-trajectory string (``num_chunk * 4 - 1``
motion steps) or an upstream trajectory JSON.

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``hy-worldplay-wan-i2v-5b``
     - Wan 2.2 TI2V-5B backbone with action + camera conditioning, PRoPE attention, and
       reconstituted-context memory. Distilled, 4 steps, streaming autoregressive VAE.

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/hy_worldplay \
       flashdreams-run \
       hy-worldplay-wan-i2v-5b \
       --help

Some generated samples from the above commands:

.. raw:: html

   <div class="model-video-grid">
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-2.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         a person walking
       </div>
     </div>
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-4.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         Walking through a seaside village
       </div>
     </div>
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-8.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         Walking through a snowy forest
       </div>
     </div>
     <div class="model-video-card">
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-9.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         Walking toward a castle
       </div>
     </div>
   </div>

Profiling benchmark
-------------------

Here is the profiling benchmark on total DiT + VAE encode/decode runtime for FlashDreams HY-WorldPlay
compared to the `official HY-WorldPlay implementation <https://github.com/Tencent-Hunyuan/HY-WorldPlay>`_
under matched settings.

.. raw:: html

   <figure class="benchmark-figure-wrap">
     <div
       id="hy-worldplay-benchmark-chart"
       class="benchmark-figure"
      data-benchmark-md-url="../_static/performance/hy_worldplay/perf-0530.md"
      data-benchmark-series="official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="HY-WorldPlay benchmark chart"
     ></div>
     <figcaption>
       <p class="model-footnote">
         This chart shows total DiT + VAE-decode runtime per autoregressive chunk (4 diffusion steps) in
         milliseconds, at steady state (median of the post-warmup chunks), measured at num_chunk=8,
         704x1280, seed=0 on a single GB300. For an apples-to-apples comparison, both implementations are
         forced to use the cuDNN attention backend and torch.compile under matched runtime settings.
         For the official HY-WorldPlay implementation, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/hy_worldplay/tests/parity_check">this instruction</a>.
       </p>
     </figcaption>
   </figure>
  <script src="../_static/js/benchmark_chart.js"></script>
