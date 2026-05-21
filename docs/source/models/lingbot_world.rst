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

Lingbot-World
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://github.com/robbyant/lingbot-world" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://github.com/robbyant/lingbot-world" target="_blank" rel="noopener noreferrer">Official code</a>
     <a class="model-link-button" href="#" target="_blank" rel="noopener noreferrer">arXiv paper (TODO)</a>
   </div>

Lingbot-World is a camera-controllable image-to-video (I2V) model with
streaming inference and context-parallel runtime support.

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/lingbot

Running the method
------------------

To run Lingbot-World, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run flashdreams-run \
       lingbot-world-fast \
       --example-data True \
       --prompt "A cinematic drone flythrough over a winding mountain road." \
       --pixel-height 464 --pixel-width 832 \
       --total-blocks 21

For multi-GPU inference, use ``torchrun`` instead of ``uv run flashdreams-run``
(taking 4 GPUs as an example):

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       lingbot-world-fast \
       --example-data True \
       --prompt "A cinematic drone flythrough over a winding mountain road." \
       --pixel-height 464 --pixel-width 832 \
       --total-blocks 21

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``lingbot-world-fast``
     - Lingbot World Fast streaming camera-control I2V (Wan VAE decoder, 4-step).
   * - ``lingbot-world-fast-flash``
     - Lingbot World Fast-Flash (LightTAE decoder, tighter streaming window).

Launch the interactive server
-----------------------------

Spin up the interactive server for Lingbot-World via webRTC:

.. code-block:: bash

   # from the repo root
   uv run --package flashdreams-lingbot python -m lingbot.webrtc.server \
       --host 0.0.0.0 --port 8089 --config_name lingbot-world-fast-flash

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

Profiling.
----------

.. figure:: /_static/perf/lingbot_total_ms.svg
   :class: benchmark-figure
   :figclass: benchmark-figure-wrap
   :alt: Lingbot-World total latency bar chart by hardware and method.

   *DiT runtime at 6-th autoregressive rollout on 4x GPUs.*
