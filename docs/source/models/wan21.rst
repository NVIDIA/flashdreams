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
     <a class="model-link-button" href="#" target="_blank" rel="noopener noreferrer">arXiv paper (TODO)</a>
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
