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

.. TODO: update code to github and arXiv link
.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/" target="_blank" rel="noopener noreferrer">Blog page</a>
     <a class="model-link-button" href="https://huggingface.co/nvidia/omni-dreams-models/" target="_blank" rel="noopener noreferrer">Model page</a>
     <a class="model-link-button" href="https://gitlab-master.nvidia.com/sil/omni-dreams/" target="_blank" rel="noopener noreferrer">Official code</a>
     <a class="model-link-button" href="#" target="_blank" rel="noopener noreferrer">arXiv paper (TODO)</a>
   </div>

As introduced in the OmniDreams project page, OmniDreams is an
HDMap-conditioned world model for single-view and multi-view driving
generation, with presets that balance visual fidelity and runtime throughput.

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
     <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
       <source src="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/teaser.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/omnidreams

Running the method
------------------

To run OmniDreams, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run flashdreams-run \
       omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae \
       --example-data True \
       --total-blocks 20

For multi-GPU inference, simply use ``uv run torchrun --nproc_per_node=4 --no-python flashdreams-run``
instead of ``uv run flashdreams-run`` (taking 4 GPUs as an example).


We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae``
     - Interactive Steering Wheel Demo Checkpoint.

.. TODO: update videos
.. raw:: html

   <div class="model-video-grid">
     <div class="model-video-card">
       <div class="model-video-placeholder">
         OmniDreams single-view sample (TODO)
       </div>
       <div class="model-video-overlay">
         Prompt1: TODO
       </div>
     </div>
     <div class="model-video-card">
       <div class="model-video-placeholder">
         OmniDreams single-view sample (TODO)
       </div>
       <div class="model-video-overlay">
         Prompt2: TODO
       </div>
     </div>
   </div>

Launch the interactive server
-----------------------------

Spin up the interactive server for OmniDreams single-view via webRTC:

.. code-block:: bash

   # from the repo root
   uv run --package flash-omnidreams torchrun --nproc_per_node 1 \
       -m omnidreams.webrtc.server \
       --host 0.0.0.0 --port 8089 \
       --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae

Then open the following URL in your browser:

- ``http://<server-ip>:8089/request_session`` to connect to the server
- ``http://<server-ip>:8089/healthz`` to check the server status (for debugging)

<server-ip> is the IP address of the server, can be "localhost" if the server is running locally.

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 0;">
     <div class="model-video-placeholder">
       Interactive server demo video placeholder
     </div>
   </div>

Performance table
-----------------

Single-view latency on NVIDIA GB300 (``704 x 1280``).

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Stage
     - 1x GPU
     - 2x GPU
     - 4x GPU
     - 8x GPU
   * - HDMap Encoder
     - 52 ms
     - 51 ms
     - 51 ms
     - 50 ms
   * - Diffusion DiT
     - 89 ms
     - 78 ms
     - 62 ms
     - 59 ms
   * - VAE Decoder
     - 13 ms
     - 13 ms
     - 14 ms
     - 13 ms
   * - KV-cache Update
     - 40 ms
     - 36 ms
     - 34 ms
     - 42 ms
   * - **Total**
     - **154 ms**
     - **143 ms**
     - **127 ms**
     - **121 ms**
   * - **Effective FPS**
     - **52**
     - **56**
     - **63**
     - **66**

*KV-cache Update is off the hot path and excluded from Total.*
