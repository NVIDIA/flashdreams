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

Causal Wan2.2
===================================
(TODO: To be updated)

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/hao-ai-lab/FastVideo" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://github.com/hao-ai-lab/FastVideo" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

This integration brings FastVideo's Causal Wan2.2 T2V variant into the
FlashDreams streaming runtime.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/fastvideo_causal_wan22

Running the method
------------------

To run Causal Wan2.2, launch the registered runner slug via
``flashdreams-run``:

.. code-block:: bash

   uv run flashdreams-run fastvideo-causal-wan2.2-t2v-14b \
       --prompt "A sports car drifting through neon rain at night." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 21

For multi-GPU inference, use ``torchrun`` on top of ``uv run flashdreams-run``
(taking 4 GPUs as an example):

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       fastvideo-causal-wan2.2-t2v-14b \
       --prompt "A sports car drifting through neon rain at night." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 21

We provide the following variant:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``fastvideo-causal-wan2.2-t2v-14b``
     - FastVideo CausalWan 2.2 14B MoE T2V (Wan VAE decoder, 8-step).

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/fastvideo_causal_wan22 flashdreams-run \
       fastvideo-causal-wan2.2-t2v-14b \
       --help
