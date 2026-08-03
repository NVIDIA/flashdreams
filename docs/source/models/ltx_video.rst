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

LTX-Video
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/Lightricks/LTX-Video" target="_blank" rel="noopener noreferrer">Official code</a>
     <a class="model-link-button" href="https://huggingface.co/Lightricks/LTX-Video" target="_blank" rel="noopener noreferrer">Model weights</a>
   </div>

`LTX-Video <https://github.com/Lightricks/LTX-Video>`_ is a 2B causal video diffusion
model with a causal VAE suited to autoregressive chunk generation. This integration
wraps ``LTXPipeline`` from ``diffusers`` and adds an optional optimized path with
manual denoising, KV-cache, ``torch.compile``, and FlashAttention.

Requirements
------------

- **Minimum VRAM**: ~24 GB for the streaming runner; ~40 GB+ for the optimized
  stack at 768×512 with KV-cache enabled.
- **PyTorch**: >= 2.9 (CUDA 13.x recommended; see :doc:`/quickstart/installation`).
- **diffusers**: recent release with ``LTXPipeline`` support.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/ltx_video

Running the method
------------------

Launch one of the registered runner slugs via ``flashdreams-run``:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run --project integrations/ltx_video \
       flashdreams-run \
       ltx-video-t2v-2b \
       --prompt "A coastal road at dusk, waves breaking on rocky cliffs, cinematic wide shot" \
       --pixel-height 512 --pixel-width 768 \
       --total-blocks 7

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``ltx-video-t2v-2b``
     - Streaming ``LTXPipeline`` wrapper — fast startup, native ``pipe()`` per chunk.
   * - ``ltx-video-t2v-2b-optimized``
     - Manual denoise loop with KV-cache, ``torch.compile``, and FlashAttention.
   * - ``ltx-video-t2v-2b-taehv``
     - Optimized path plus TAEHV fast decoder.

The optimized runner improves time-to-first-frame after warmup; steady-state
throughput tuning is ongoing.

Tests
-----

.. code-block:: bash

   uv run --project integrations/ltx_video pytest integrations/ltx_video/tests/test_smoke.py -v

GPU optimization tests (requires CUDA + weights):

.. code-block:: bash

   uv run --project integrations/ltx_video pytest integrations/ltx_video/tests/test_optimizations.py -v -m ci_gpu
