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

LingBot-World
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://technology.robbyant.com/lingbot-world" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://github.com/robbyant/lingbot-world" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

As introduced by `Robbyant <https://technology.robbyant.com/>`_, LingBot-World is a camera-controllable image-to-video
(I2V) world model with streaming inference and context-parallel runtime support.

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
     <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
       <source src="https://gw.alipayobjects.com/v/huamei_u94ywh/afts/video/XQk7Rb44qJwAAAAAgfAAAAgAfoeUAQBr" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/lingbot

Running the method
------------------

To run LingBot-World, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run --project integrations/lingbot \
       flashdreams-run \
       lingbot-world-fast \
       --example-data True \
       --pixel-height 464 --pixel-width 832 \
       --total-blocks 21

For multi-GPU inference, use ``torchrun`` instead of ``uv run flashdreams-run``
(taking 4 GPUs as an example):

.. code-block:: bash

   uv run --project integrations/lingbot \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       lingbot-world-fast \
       --example-data True \
       --pixel-height 464 --pixel-width 832 \
       --total-blocks 21

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``lingbot-world-fast``
     - Official camera-control I2V (Wan VAE decoder, full KV-cache).
   * - ``lingbot-world-fast-flash``
     - Efficient streaming configuration (TAEHV decoder, window + sink KV-cache).

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/lingbot \
       flashdreams-run \
       lingbot-world-fast \
       --help

Some generated samples:

.. raw:: html

   <div class="model-video-grid">
     <div class="model-video-card">
       <div class="model-video-placeholder">Video placeholder</div>
       <!-- <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research-staging.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         example_data_uuid: "239560dc-33d1-11ef-9720-00044bcbccac"
       </div> -->
     </div>
     <div class="model-video-card">
       <div class="model-video-placeholder">Video placeholder</div>
       <!-- <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research-staging.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-24b84744-4156-11ef-b27d-00044bf655de-pip.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         example_data_uuid: "24b84744-4156-11ef-b27d-00044bf655de"
       </div> -->
     </div>
   </div>


Launch the interactive server
-----------------------------

Spin up the interactive LingBot-World server via WebRTC:

.. code-block:: bash

   # from the repo root
   uv run --package flashdreams-lingbot torchrun --nproc_per_node 1 \
       -m lingbot.webrtc.server \
       --host 0.0.0.0 --port 8089 \
       --config_name lingbot-world-fast-flash

The server may take a few minutes to warm up. When it is ready, it prints
``Connect via http://<server-ip>:8089/request_session``.
Here, ``<server-ip>`` is the server IP address you are connecting to
(can use ``localhost`` when running locally).

When successfully connected, the browser-based UI looks like this:


When successfully connected, the browser-based UI looks like this:


.. raw:: html

  <div class="model-video-card">
    <div class="model-video-placeholder">Recorded interactive serving demo (placeholder)</div>
  </div>

Profiling Benchmark
-------------------

Here is the profiling benchmark on total DiT runtime for FlashDreams LingBot-World
compared to the `official LingBot-World implementation <https://github.com/robbyant/lingbot-world>`_
and `LightX2V <https://github.com/ModelTC/lightx2v>`_ under
matched settings.

.. raw:: html

   <figure class="benchmark-figure-wrap">
     <div
       id="lingbot-world-benchmark-chart"
       class="benchmark-figure"
       data-benchmark-md-url="/_static/performance/lingbot_world/perf-0521.md"
      data-benchmark-series="official:Official Impl:#3b82f6;lightx2v:LightX2V:#f59e0b;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="LingBot-World benchmark chart"
     ></div>
     <figcaption>
       <p>
         This chart shows total DiT runtime (4 diffusion steps) in milliseconds at the 6th autoregressive rollout on 4x GPUs.
         For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend under matched runtime settings,
         and all runs use Ulysses sequence parallelism for multi-GPU inference.
         For the official LingBot-World implementation, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot/tests/parity_check">this instruction</a>.
         For the LightX2V baseline, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot/tests/baseline_lightx2v">this instruction</a>.
       </p>
     </figcaption>
   </figure>
   <script src="/_static/js/benchmark_chart.js"></script>
