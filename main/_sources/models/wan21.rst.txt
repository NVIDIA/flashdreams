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

Wan2.1
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/Wan-Video/Wan2.1" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://github.com/Wan-Video/Wan2.1" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Wan2.1 is the bidirectional reference family in FlashDreams, supporting both
text-to-video (T2V) and image-to-video (I2V) inference presets.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/wan21

Running the method
------------------

To run Wan2.1, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run flashdreams-run wan21-t2v-1.3b-480p \
       --prompt "A reindeer in cinematic sunset light." \
       --pixel-height 480 --pixel-width 832

For multi-GPU inference, use ``torchrun`` instead of ``uv run flashdreams-run``
(taking 4 GPUs as an example):

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       wan21-t2v-1.3b-480p \
       --prompt "A reindeer in cinematic sunset light." \
       --pixel-height 480 --pixel-width 832

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``wan21-t2v-1.3b-480p``
     - Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only).
   * - ``wan21-i2v-14b-480p``
     - Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame).

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/wan21 flashdreams-run \
       wan21-t2v-1.3b-480p \
       --help

Profiling Benchmark
-------------------

Here is the profiling benchmark on DiT per-step runtime for FlashDreams Wan2.1
compared to the official Wan2.1 implementation and the FastVideo baseline under
matched settings.

.. raw:: html

   <figure class="benchmark-figure-wrap">
     <div
       id="wan21-benchmark-chart"
       class="benchmark-figure"
       data-benchmark-md-url="/_static/performance/wan21/perf-0521.md"
       data-benchmark-series="fastvideo:FastVideo:#f59e0b;official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="Wan2.1 benchmark chart"
     ></div>
     <figcaption>
       <p>
         This chart shows per-diffusion-step DiT runtime in milliseconds with CFG at 480p (81 frames) on a single GPU.
         For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend under matched runtime settings.
         For the official Wan2.1 implementation, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/wan21/tests/parity_check">this instruction</a>.
         For the FastVideo baseline, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/wan21/tests/baseline_fastvideo">this instruction</a>.
       </p>
     </figcaption>
   </figure>
   <script src="/_static/js/benchmark_chart.js"></script>
