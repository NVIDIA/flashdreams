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

Self-Forcing
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://self-forcing.github.io/" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://arxiv.org/abs/2506.08009" target="_blank" rel="noopener noreferrer">arXiv paper</a>
     <a class="model-link-button" href="https://github.com/guandeh17/Self-Forcing" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Self-Forcing is a text-to-video (T2V) model based on :doc:`Wan2.1 </models/wan21>`.
It uses a training paradigm for autoregressive video diffusion that simulates
inference-time rollout during training with KV caching, reducing the train-test
gap and enabling efficient streaming generation quality.

.. image:: https://self-forcing.github.io/static/teaser.jpg
   :alt: Self-Forcing teaser figure.
   :width: 100%

.. raw:: html

   <p class="model-footnote">
     Teaser image source:
     <a href="https://self-forcing.github.io/">Self-Forcing project page</a>.
   </p>

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/self_forcing

Running the method
------------------

To run Self-Forcing, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run --project integrations/self_forcing \
       flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b \
       --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 7

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``self-forcing-wan2.1-t2v-1.3b``
     - Official checkpoint.
   * - ``self-forcing-wan2.1-t2v-1.3b-taehv``
     - Official checkpoint. Swap Wan VAE decoder with the faster TAEHV decoder.
   * - ``self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope``
     - Steady long-rollout preset: static sink=5 + rolling window=7, with
       KVCache-relative RoPE.

For multi-GPU inference, use:

.. code-block:: bash

   uv run --project integrations/self_forcing \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b \
       --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 7

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/self_forcing \
       flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b \
       --help

What to expect
--------------

**Default prompt.** Omitting ``--prompt`` uses a Tokyo street-scene
default (the same prompt shown in the command above). To override, pass
either an inline string or a path to a ``.txt`` file whose first
non-empty line is read as the prompt.

**Total blocks.** ``--total-blocks N`` runs ``N`` autoregressive chunks
of the streaming pipeline; each chunk emits a fixed number of decoded
video frames, so longer rollouts come from larger ``--total-blocks``.
The commands on this page use ``--total-blocks 7`` to keep the demo
fast; the runner config's own default is ``60`` for full-length
rollouts. See :doc:`/developer_guides/inference_pipeline_overview` for
what one chunk maps to in the encoder / diffusion / decoder loop.

**Outputs.** The runner writes to ``outputs/`` (relative to the current
working directory) by default:

- ``outputs/<runner-slug>.mp4`` — the generated video at 16 FPS, 480×832
  for the defaults above (override resolution with ``--pixel-height`` /
  ``--pixel-width`` and frame rate with ``--fps``).
- ``outputs/stats_<runner-slug>.json`` — per-autoregressive-step timing
  stats, when the pipeline reports them.

Pass ``--output-dir <path>`` to redirect the destination.

**Runtime (measured on H100 80GB, ``--total-blocks 7``, defaults
otherwise).** First runs are dominated by Triton autotuning and
CUDA-graph warmup; subsequent runs in the same workspace reuse cached
kernels and are much faster:

.. list-table::
   :header-rows: 1
   :widths: 36 32 32

   * - Setup
     - First run (cold)
     - Subsequent runs
   * - 1× H100 PCIe
     - ~6.9 min
     - ~42 s
   * - 4× H100 HBM3 (``torchrun --nproc_per_node=4``)
     - ~8.6 min
     - ~73 s

Most of the cold-run cost lands in the first two AR blocks; steady-state
blocks (AR 2 onward) are sub-second.

**When to use multi-GPU.** Per-block steady-state cost on 4 GPUs is
roughly 2× faster than on 1 GPU (~251 ms vs ~500 ms), but the larger
per-rank autotune + NCCL overhead makes 4 GPUs ~24% *slower* end-to-end
for short rollouts like ``--total-blocks 7``. Multi-GPU only pays off
once steady-state dominates warmup — stay on 1 GPU for short demos and
move to ``torchrun --nproc_per_node=4`` for full-length rollouts
(``--total-blocks 60+``).

Some generated samples from the above commands:

.. raw:: html

   <div class="model-video-grid">
     <div class="model-video-card">
       <!-- <div class="model-video-placeholder">Video placeholder</div> -->
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash_1.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         A close-up shot of a ceramic teacup slowly pouring water into a glass mug. The water flows smoothly from the spout of the teacup into the mug, creating gentle ripples as it fills up. Both cups have detailed textures, with the teacup having a matte finish and the glass mug showcasing clear transparency. The background is a blurred kitchen countertop, adding context without distracting from the central action. The pouring motion is fluid and natural, emphasizing the interaction between the two cups.
       </div>
     </div>
     <div class="model-video-card">
       <!-- <div class="model-video-placeholder">Video placeholder</div> -->
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash_6.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         A dramatic and dynamic scene in the style of a disaster movie, depicting a powerful tsunami rushing through a narrow alley in Bulgaria. The water is turbulent and chaotic, with waves crashing violently against the walls and buildings on either side. The alley is lined with old, weathered houses, their facades partially submerged and splintered. The camera angle is low, capturing the full force of the tsunami as it surges forward, creating a sense of urgency and danger. People can be seen running frantically, adding to the chaos. The background features a distant horizon, hinting at the larger scale of the tsunami. A dynamic, sweeping shot from a low-angle perspective, emphasizing the movement and intensity of the event.
       </div>
     </div>
   </div>


Profiling benchmark
-------------------

Here is the profiling benchmark on total DiT runtime for FlashDreams Self-Forcing compared to
the `official Self-Forcing implementation <https://github.com/guandeh17/Self-Forcing>`_
and the `FastVideo implementation <https://github.com/hao-ai-lab/FastVideo>`_
under matched settings.

.. raw:: html

   <figure class="benchmark-figure-wrap">
     <div
       id="self-forcing-benchmark-chart"
       class="benchmark-figure"
      data-benchmark-md-url="../_static/performance/self_forcing/perf-0521.md"
       data-benchmark-series="fastvideo:FastVideo:#f59e0b;official:Official Impl:#3b82f6;flashdreams:FlashDreams:#76B900"
       data-chart-aria-label="Self-Forcing benchmark chart"
     ></div>
     <figcaption>
      <p class="model-footnote">
         This chart shows the DiT total runtime (4 denoising steps in milliseconds) at the 6th autoregressive rollout on a single GPU.
         For an apples-to-apples comparison, all implementations are forced to use cuDNN attention backend and <code>torch.compile</code> for DiT network.
         For profiling the official implementation, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing/tests/parity_check/README.md">this instruction</a>.
         For profiling the FastVideo implementation, see
         <a href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing/tests/baseline_fastvideo/README.md">this instruction</a>.
       </p>
     </figcaption>
   </figure>
  <script src="../_static/js/benchmark_chart.js"></script>
