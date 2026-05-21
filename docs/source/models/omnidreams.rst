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

OmniDreams
===================================

Overview
--------

OmniDreams in FlashDreams targets HDMap-conditioned world-model generation for
single-view and multi-view driving scenarios, with optimized presets for both
quality and throughput.

Links
-----

- Project page: `Omni Dreams models on Hugging Face <https://huggingface.co/nvidia/omni-dreams-models>`_
- Sample data: `Omni Dreams samples <https://huggingface.co/datasets/nvidia/omni-dreams-samples>`_
- Integration package: `flashdreams/integrations/omnidreams <https://github.com/NVIDIA/flashdreams/tree/main/integrations/omnidreams>`_

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
     <strong>Qualitative outputs and benchmark snapshots</strong><br>
     <a href="https://github.com/user-attachments/assets/94be56d9-2d89-4691-90c4-95faf5c02fe7" target="_blank" rel="noopener noreferrer">
       Open OmniDreams results asset (GitHub hosted)
     </a>
     <br>
     Inline playback requires a direct public media URL (for example, a ``.mp4`` URL from a public CDN).
   </div>

Real-time serving recording
---------------------------

.. raw:: html

   <div class="video-slot">
     <strong>Serving demo</strong><br>
     WebRTC serving launch and runtime details are documented in
     ``integrations/omnidreams/README.md``.
   </div>
