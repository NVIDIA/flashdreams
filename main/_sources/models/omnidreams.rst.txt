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

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/" target="_blank" rel="noopener noreferrer">Blog page</a>
     <a class="model-link-button" href="https://huggingface.co/nvidia/omni-dreams-models/" target="_blank" rel="noopener noreferrer">Model page</a>
     <a class="model-link-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/omnidreams" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

OmniDreams is an HDMap-conditioned world model for single-view and multi-view
driving generation, with presets that balance visual fidelity and runtime
throughput.

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
     <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
       <source src="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/teaser.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>
   <p class="model-footnote">
     Teaser video source:
     <a href="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/">OmniDreams project page</a>.
   </p>

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

   uv run --project integrations/omnidreams \
       flashdreams-run \
       omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --example-data True \
       --example_data_uuid "239560dc-33d1-11ef-9720-00044bcbccac" \
       --total-blocks 20

Sample example-data UUIDs for the inference script are available in the
`nvidia/omni-dreams-samples Hugging Face dataset <https://huggingface.co/datasets/nvidia/omni-dreams-samples/tree/main/data/single_view>`_.

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf``
     - Single-view 2-step HDMap-conditioned I2V.

For multi-GPU inference, use:

.. code-block:: bash

   uv run --project integrations/omnidreams \
       torchrun --nproc_per_node=4 --no-python flashdreams-run \
       omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --example-data True \
       --example_data_uuid "239560dc-33d1-11ef-9720-00044bcbccac" \
       --total-blocks 20

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/omnidreams \
       flashdreams-run \
       omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --help

Some generated samples from the above commands:

.. raw:: html

   <div class="model-video-grid">
     <div class="model-video-card">
       <!-- <div class="model-video-placeholder">Video placeholder</div> -->
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         example_data_uuid: "239560dc-33d1-11ef-9720-00044bcbccac"
       </div>
     </div>
     <div class="model-video-card">
       <!-- <div class="model-video-placeholder">Video placeholder</div> -->
       <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
         <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-24b84744-4156-11ef-b27d-00044bf655de-pip.mp4" type="video/mp4">
         Your browser does not support the video tag.
       </video>
       <div class="model-video-overlay">
         example_data_uuid: "24b84744-4156-11ef-b27d-00044bf655de"
       </div>
     </div>
   </div>

Launch the interactive demo
---------------------------

``interactive-drive`` runs the OmniDreams single-view pipeline in a
single process and streams the camera view to your browser. The demo
machine only needs a CUDA-capable GPU -- no graphics-capable GPU,
display server, or Vulkan toolchain required.

Requires access to `NVIDIA/flashdreams <https://github.com/NVIDIA/flashdreams>`_
and an ``HF_TOKEN`` with read access to
`nvidia/omni-dreams-scenes <https://huggingface.co/datasets/nvidia/omni-dreams-scenes>`_
(scene USDZs) and
`nvidia/omni-dreams-models <https://huggingface.co/nvidia/omni-dreams-models>`_
(world-model checkpoints).

First-time setup:

.. code-block:: bash

   git clone git@github.com:NVIDIA/flashdreams.git
   cd flashdreams
   export HF_TOKEN=<your-hf-token>
   uv sync --package flashdreams-omnidreams --extra interactive-drive

Optionally pre-download scenes and checkpoints so the first launch
isn't blocked on network I/O:

.. code-block:: bash

   uv run --package flashdreams-omnidreams omnidreams-prepare

Run the demo and stream to your browser:

.. code-block:: bash

   uv run --package flashdreams-omnidreams interactive-drive --stream-mjpeg :8080

Then open ``http://<server-ip>:8080/`` in any browser on the same
network and pick a scene from the picker in the bottom-right.

For deployments with a desktop NVIDIA GPU that has a graphics queue,
omit ``--stream-mjpeg`` to open the demo in a local Vulkan window
instead:

.. code-block:: bash

   uv run --package flashdreams-omnidreams interactive-drive

Alternative: WebRTC server
--------------------------

For deployments that need a richer browser frontend with WebRTC's
lower video-delivery latency and a streaming gRPC service for
multi-client setups, the standalone server at
``omnidreams.webrtc.server`` ships a polished HTML5 client on top of
the same OmniDreams pipeline. The MJPEG path above is the
recommended starting point for most users; reach for WebRTC when you
need bidirectional camera-control APIs or are already integrating
the gRPC service into a larger product.

.. code-block:: bash

   # from the repo root
   uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 \
       -m omnidreams.webrtc.server \
       --host 0.0.0.0 --port 8089 \
       --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --scene-uuid "065dcac9-ee67-4434-a835-c6b816c88e48"

Sample scene UUIDs for the interactive server are available in the
`nvidia/omni-dreams-scenes Hugging Face dataset <https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes>`_.

The server may take a few minutes to warm up. When it is ready, it prints
``Connect via http://<server-ip>:8089/request_session``.
Here, ``<server-ip>`` is the server IP address you are connecting to
(can use ``localhost`` when running locally).

When successfully connected, the browser-based UI looks like this:

.. raw:: html

  <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
    <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
      <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-webrtc-recording-0529.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>

.. note::

   If ``/request_session`` loads but the video never appears, the
   browser is likely obfuscating local IPs in WebRTC ICE candidates
   (replacing them with mDNS ``.local`` hostnames), which prevents the
   peer connection from completing. Disable the setting and reload:

   - **Chrome / Edge:** ``chrome://flags/#enable-webrtc-hide-local-ips-with-mdns`` → **Disabled**, then restart the browser.
   - **Brave:** ``brave://settings/privacy/security`` → *WebRTC IP handling policy* → **Default public and private interfaces**.
   - **Firefox:** ``about:config`` → ``media.peerconnection.ice.obfuscate_host_addresses`` → **false**.

Performance table
-----------------

Single-view latency on NVIDIA GB300 at ``704 x 1280`` resolution.

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Stage
     - 1x GPU
     - 2x GPU
     - 4x GPU
     - 8x GPU
   * - HDMap Encoder
     - 28 ms
     - 26 ms
     - 26 ms
     - 26 ms
   * - Diffusion DiT
     - 84 ms
     - 71 ms
     - 49 ms
     - 47 ms
   * - VAE Decoder
     - 6 ms
     - 5 ms
     - 5 ms
     - 5 ms
   * - KV-cache Update
     - 42 ms
     - 34 ms
     - 23 ms
     - 22 ms
   * - **Total**
     - **118 ms**
     - **102 ms**
     - **80 ms**
     - **78 ms**
   * - **Effective FPS**
     - **68**
     - **78**
     - **100**
     - **103**

.. raw:: html

   <p class="model-footnote">
      KV-cache Update is off the hot path and excluded from Total.
   </p>
