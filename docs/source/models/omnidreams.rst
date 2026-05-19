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

Omnidreams
===================================

Overview
--------

Omnidreams in FlashDreams targets HDMap-conditioned world-model generation for
single-view and multi-view driving scenarios, with optimized presets for both
quality and throughput.

Links
-----

- Project page: `Omni Dreams models on Hugging Face <https://huggingface.co/nvidia/omni-dreams-models>`_
- Sample data: `Omni Dreams samples <https://huggingface.co/datasets/nvidia/omni-dreams-samples>`_
- Paper link (placeholder): ``TODO: add arXiv link``

Model figure
------------

.. raw:: html

   <div class="video-slot">
     <strong>Figure Placeholder</strong><br>
     Add architecture/system figure from the Omnidreams paper/project page.
   </div>

Run this model
--------------

Single-view performance preset:

.. code-block:: bash

   uv run flashdreams-run \
       omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --example-data True --total-blocks 20

4-camera multi-view:

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae \
       --example-data True --total-blocks 20

Results
-------

.. raw:: html

   <div class="video-slot">
     <strong>Results Placeholder</strong><br>
     Add side-by-side qualitative outputs and benchmark snapshots.
   </div>

Real-time serving recording
---------------------------

.. raw:: html

   <div class="video-slot">
     <strong>Serving Recording Placeholder</strong><br>
     Add screen recording of real-time Omnidreams serving here.
   </div>

For more model pages, see :doc:`/models/index`.
