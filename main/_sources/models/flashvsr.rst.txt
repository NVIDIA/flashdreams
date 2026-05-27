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
(TODO: To be updated)

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/OpenImagingLab/FlashVSR" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://github.com/OpenImagingLab/FlashVSR" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

FlashVSR is a streaming video super-resolution integration. It is useful for
workflows that ingest low-resolution video in chunks and need a higher
resolution output stream while preserving temporal continuity.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/flashvsr

Running the method
------------------

To run FlashVSR, provide an input video path and launch one of the registered
runner slugs via ``flashdreams-run``:

.. code-block:: bash

   uv run flashdreams-run \
       flashvsr-v1.1-sparse-ratio-2.0 \
       --input-path /path/to/low_res_input.mp4 \
       --chunk-size 16

For multi-GPU inference, use the full-attention preset:

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       flashvsr-v1.1-full-attn \
       --input-path /path/to/low_res_input.mp4 \
       --chunk-size 16

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

   uv run --project integrations/flashvsr flashdreams-run \
       flashvsr-v1.1-sparse-ratio-2.0 \
       --help

What FlashDreams accelerates
----------------------------

FlashDreams keeps the video super-resolution pipeline chunk-oriented: the
runner reads input video, derives per-video dimensions, initializes streaming
state, and then runs autoregressive chunks through the pipeline. The
full-attention preset can also use context-parallel execution for heavier
multi-GPU runs.
