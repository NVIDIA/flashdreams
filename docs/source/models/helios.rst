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

Helios
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/PKU-YuanGroup/Helios" target="_blank" rel="noopener noreferrer">Official code</a>
     <a class="model-link-button" href="https://huggingface.co/BestWishYsh/Helios-Distilled" target="_blank" rel="noopener noreferrer">Helios-Distilled</a>
     <a class="model-link-button" href="https://huggingface.co/BestWishYsh/Helios-Base" target="_blank" rel="noopener noreferrer">Helios-Base</a>
   </div>

`Helios <https://github.com/PKU-YuanGroup/Helios>`_ is a 14B real-time streaming
text-to-video model. This integration wraps ``HeliosPyramidPipeline`` from
``diffusers`` and exposes Helios' native **33-frame chunks** through the
FlashDreams streaming ``generate()`` interface.

Requirements
------------

- **Minimum VRAM**: ~80 GB (14B transformer + VAE; tested on H100 80GB).
- **PyTorch**: >= 2.9 (CUDA 13.x recommended; see :doc:`/quickstart/installation`).
- **diffusers**: ``HeliosPyramidPipeline`` requires a recent ``diffusers`` build
  (install from source if the PyPI release on your platform does not yet export it):

  .. code-block:: bash

     pip install git+https://github.com/huggingface/diffusers.git

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/helios

Running the method
------------------

Launch one of the registered runner slugs via ``flashdreams-run``:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run --project integrations/helios \
       flashdreams-run \
       helios-distilled-t2v-14b \
       --prompt "A coastal road at dusk, waves breaking on rocky cliffs, cinematic wide shot" \
       --pixel-height 384 --pixel-width 640 \
       --total-blocks 3

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``helios-distilled-t2v-14b``
     - ``BestWishYsh/Helios-Distilled`` — fastest inference (pyramid ``[2,2,2]``, no CFG).
   * - ``helios-base-t2v-14b``
     - ``BestWishYsh/Helios-Base`` — highest quality (pyramid ``[20,20,20]``, CFG 5.0).
   * - ``helios-distilled-t2v-14b-2gpu``
     - Distilled checkpoint with Ulysses context parallelism (``torchrun``, 2+ GPUs).

Multi-GPU (2× H100)
-------------------

.. code-block:: bash

   torchrun --nproc_per_node=2 --no-python \
       flashdreams-run helios-distilled-t2v-14b-2gpu \
       --total-blocks 8

Each ``generate()`` call produces one 33-frame chunk and yields decoded pixels
immediately, matching Helios' native streaming cadence.

Tests
-----

.. code-block:: bash

   uv run pytest integrations/helios/tests/test_smoke.py -v
