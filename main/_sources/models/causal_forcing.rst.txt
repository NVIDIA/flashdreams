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

Causal-Forcing
===================================
(TODO: To be updated)

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/LiRunyi2001/causal-forcing" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://github.com/LiRunyi2001/causal-forcing" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Causal-Forcing provides streaming :doc:`Wan2.1 </models/wan21>` variants for
both text-to-video (T2V) and image-to-video (I2V) generation.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/causal_forcing

Running the method
------------------

To run Causal-Forcing, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run flashdreams-run \
       causal-forcing-wan2.1-t2v-1.3b-framewise \
       --prompt "A cat surfing on a neon wave." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 21

For multi-GPU inference, use ``torchrun`` on top of ``uv run flashdreams-run``
(taking 4 GPUs as an example):

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       causal-forcing-wan2.1-t2v-1.3b-framewise \
       --prompt "A cat surfing on a neon wave." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 21

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``causal-forcing-wan2.1-t2v-1.3b-chunkwise``
     - Causal-Forcing chunkwise Wan 2.1 1.3B T2V (``len_t=3``).
   * - ``causal-forcing-wan2.1-t2v-1.3b-framewise``
     - Causal-Forcing framewise Wan 2.1 1.3B T2V (``len_t=1``).
   * - ``causal-forcing-wan2.1-i2v-1.3b-framewise``
     - Causal-Forcing framewise Wan 2.1 1.3B I2V (``len_t=1``).

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/causal_forcing flashdreams-run \
       causal-forcing-wan2.1-t2v-1.3b-framewise \
       --help
