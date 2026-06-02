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

NVIDIA OmniDreams
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://research.nvidia.com/labs/sil/projects/omnidreams-blog/" target="_blank" rel="noopener noreferrer">Blog page</a>
     <a class="model-link-button" href="https://huggingface.co/nvidia/omni-dreams-models/" target="_blank" rel="noopener noreferrer">Model page</a>
     <a class="model-link-button" href="https://github.com/NVIDIA/flashdreams/tree/main/integrations/omnidreams" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

OmniDreams is a HDMap-conditioned world model for single-view and multi-view
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

   <div class="model-video-grid zoomable">
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
display server, or Vulkan support are required.

The demo requires access to `NVIDIA/flashdreams <https://github.com/NVIDIA/flashdreams>`_
and an ``HF_TOKEN`` with read access to
`nvidia/omni-dreams-scenes <https://huggingface.co/datasets/nvidia/omni-dreams-scenes>`_
(scene USDZs) and
`nvidia/omni-dreams-models <https://huggingface.co/nvidia/omni-dreams-models>`_
(checkpoints).

First-time setup:

.. code-block:: bash

   git clone https://github.com/NVIDIA/flashdreams.git
   cd flashdreams
   export HF_TOKEN=<your-hf-token>
   uv sync --package flashdreams-omnidreams --extra interactive-drive

Optionally, pre-download scenes and checkpoints so the first launch
isn't blocked on network I/O:

.. code-block:: bash

   uv run --package flashdreams-omnidreams omnidreams-prepare

Run the demo and stream to your browser:

.. code-block:: bash

   uv run --package flashdreams-omnidreams interactive-drive --stream-mjpeg :8080

Then open ``http://<server-ip>:8080/`` in any browser on the same
network and pick a scene from the picker in the bottom-right.

For execution using a consumer NVIDIA GPU that exposes a graphics stack,
omit the ``--stream-mjpeg`` flag to open the demo in a local Vulkan window
instead:

.. code-block:: bash

   uv run --package flashdreams-omnidreams interactive-drive

The local window's HUD adds a weather-variant selector (clear, rain, snow)
next to the scene picker, so the same scene can be switched between
conditions.

.. note::

   The local window requires a display server and the system OpenGL /
   Vulkan client libraries. On Debian/Ubuntu:

   .. code-block:: bash

      sudo apt install -y libx11-6 libxcb1 libgl1 libglx-mesa0 libvulkan1

   A ``Failed to initialize GLFW`` error indicates the display or one of these
   libraries are missing.

Steering wheel and game controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A steering wheel or game controller can be used to control the local window mode.
Any device that Ubuntu detects as a standard game controller
or joystick is viable. We provide a configuration tool to calibrate these:

.. code-block:: bash

   uv run --package flashdreams-omnidreams interactive-drive-configuration

The demo loads the saved profile automatically on subsequent launches.
Re-run the configuration tool to specify the default profile, edit a profile
(steering sensitivity, deadzone, buttons), or delete a profile.

Native acceleration (perf manifest)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bundled ``example_world_model_perf.yaml`` manifest runs the DiT and
LightVAE through the OmniDreams single-view CUDA extension
(``native_dit_acceleration: required``), which is faster than the default
PyTorch path. The extension builds against pinned checkouts of CUTLASS,
SageAttention, SpargeAttn, and cudnn-frontend that are not vendored in the
repo. ``omnidreams-prepare --perf`` clones them at their pinned commits into
``integrations/omnidreams/omnidreams_singleview/3rdparty/``:

.. code-block:: bash

   uv run --package flashdreams-omnidreams omnidreams-prepare --perf

This step only syncs sources; the extension itself compiles on the first
launch that uses the manifest (one-time, a few minutes). It requires a
Blackwell-class GPU (SM 12.0) or newer, a source checkout (the
``omnidreams_singleview`` sources ship only in the git tree, not the wheel),
``git``, and a CUDA toolchain (``nvcc``) matching your PyTorch build. Then
point the demo at the perf manifest:

.. code-block:: bash

   uv run --package flashdreams-omnidreams interactive-drive \
       --manifest example_world_model_perf.yaml

``native_dit_acceleration: required`` makes the manifest fail loudly if the
extension can't build or load, rather than silently falling back to PyTorch.

Alternative: WebRTC server
--------------------------

For deployments that require a richer browser frontend with WebRTC's
lower video-delivery latency and a streaming gRPC service for
multi-client setups, the standalone server at
``omnidreams.webrtc.server`` ships a polished HTML5 client on top of
the same OmniDreams pipeline. The MJPEG path above is the
recommended starting point for most users; consider WebRTC if you
need bidirectional camera-control APIs or are already integrating
the gRPC service into a larger product.

.. code-block:: bash

   # from the repo root
   uv run --package flashdreams-omnidreams torchrun --nproc_per_node 1 \
       -m omnidreams.webrtc.server \
       --host 0.0.0.0 --port 8089 \
       --pipeline_config_name omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --scene-uuid "0d404ff7-2b66-498c-b047-1ed8cded60d4"

Sample scene UUIDs for the interactive server are available in the
`nvidia/omni-dreams-scenes Hugging Face dataset <https://huggingface.co/datasets/nvidia/omni-dreams-scenes/tree/main/scenes>`_.
Each scene ships clear, rain, and snow weather variants as sibling
archives; add ``--scene-variant rain`` (or ``snow``) to serve a specific
one (the default is the clear-weather scene).

The server may take a few minutes to warm up. Once ready, it prints
``Connect via http://<server-ip>:8089/request_session``.
Here, ``<server-ip>`` is the server IP address you are connecting to
(can use ``localhost`` when running locally).

.. note::

   On a remote or cloud GPU instance (e.g. `Brev <https://www.brev.dev/>`_),
   the server port is usually not reachable at the host IP directly.
   Forward it to your local machine first, then open
   ``http://localhost:8089/request_session``:

   .. code-block:: bash

      # Brev
      brev port-forward <instance> -p 8089:8089
      # or plain SSH
      ssh -L 8089:localhost:8089 <user>@<host>

Once successfully connected, the browser-based UI looks like this:

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
