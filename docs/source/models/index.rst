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

Models
======

.. toctree::
   :hidden:
   :maxdepth: 1

   omnidreams
   self_forcing
   causal_forcing
   causal_wan22
   cosmos_predict2
   flashvsr
   hy_worldplay
   lingbot_world
   wan21

FlashDreams runs a growing family of world and video models (text-to-video,
image-to-video, camera-controlled, and super-resolution), all through one
consistent command line and Python interface. Browse the models below, pick the
one that fits what you want to make, and follow its card through to the full
method.

Available models
----------------

The models come in three flavors. Streaming and autoregressive generation
methods build a video step by step and stay fast once warmed up, aiming for
sub-second latency per step; bidirectional methods produce a clip in a single
pass and serve as the quality reference for their streaming counterparts; and
super-resolution methods upscale existing frames in chunks, so their latency
scales with output resolution rather than step count. Each card links to that
method's page, where you'll find the exact command to run it, the checkpoint it
uses, and the settings you can tune.

.. container:: fd-eyebrow

   Streaming and autoregressive generation

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: OmniDreams
      :class-card: fd-feature
      :link: /models/omnidreams
      :link-type: doc

      .. raw:: html

         <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
           <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.mp4" type="video/mp4">
         </video>

      Interactive world simulator for autonomous vehicles.

   .. grid-item-card:: Self-Forcing
      :class-card: fd-feature
      :link: /models/self_forcing
      :link-type: doc

      .. raw:: html

         <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
           <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash_1.mp4" type="video/mp4">
         </video>

      Autoregressive text-to-video based on Wan 2.1.

   .. grid-item-card:: Causal-Forcing
      :class-card: fd-feature
      :link: /models/causal_forcing
      :link-type: doc

      .. raw:: html

         <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
           <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_forcing/causal-forcing-wan2.1-t2v-1.3b-framewise.mp4" type="video/mp4">
         </video>

      Autoregressive text/image-to-video based on Wan 2.1.

   .. grid-item-card:: Causal Wan 2.2
      :class-card: fd-feature
      :link: /models/causal_wan22
      :link-type: doc

      .. raw:: html

         <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
           <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_wan22/fastvideo-causal-wan2.2-t2v-14b_1.mp4" type="video/mp4">
         </video>

      Autoregressive text-to-video based on Wan 2.2 from FastVideo.

   .. grid-item-card:: LingBot-World
      :class-card: fd-feature
      :link: /models/lingbot_world
      :link-type: doc

      .. raw:: html

         <div class="fd-card-video-wrap">
           <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
             <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-01.mp4" type="video/mp4">
           </video>
           <video class="fd-card-video-pip" autoplay muted loop playsinline preload="metadata">
             <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-traj-01.mp4" type="video/mp4">
           </video>
         </div>

      Camera-controllable image-to-video world model.

   .. grid-item-card:: HY-WorldPlay
      :class-card: fd-feature
      :link: /models/hy_worldplay
      :link-type: doc

      .. raw:: html

         <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
           <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-hero.mp4" type="video/mp4">
         </video>

      Action- and camera-controllable image-to-video world model.

.. container:: fd-eyebrow

   Bidirectional Video Generation

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Wan 2.1
      :class-card: fd-feature
      :link: /models/wan21
      :link-type: doc

      .. raw:: html

         <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
           <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/wan21/wan21-t2v-1.3b-480p.mp4" type="video/mp4">
         </video>

      Bidirectional video generation model that supports both
      text-to-video and image-to-video.

   .. grid-item-card:: Cosmos-Predict2.5
      :class-card: fd-feature
      :link: /models/cosmos_predict2
      :link-type: doc

      .. raw:: html

         <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
           <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/cosmos_predict2/cosmos2-t2v-2b-720p.mp4" type="video/mp4">
         </video>

      Bidirectional Cosmos-Predict2 reference implementations (T2V / I2V, 2B).

.. container:: fd-eyebrow

   Super-resolution

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: FlashVSR
      :class-card: fd-feature
      :link: /models/flashvsr
      :link-type: doc

      .. raw:: html

         <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
           <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/flashvsr/flashvsr-v1.1-sparse-ratio-2.0.mp4" type="video/mp4">
         </video>

      Streaming video super-resolution.

Running a model yourself
------------------------

.. code-block:: bash

   uv run flashdreams-run <MODEL_SLUG> --help

Examples:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-taehv --total-blocks 7
   uv run flashdreams-run lingbot-world-fast --example-data True --total-blocks 21

Adding your own model
~~~~~~~~~~~~~~~~~~~~~~

See :doc:`/developer_guides/new_integration` for model integration and registration
guidance.

Related
-------

- Follow the :doc:`/quickstart/index` for the shortest path to
  running these methods on your own hardware.
- The :doc:`/developer_guides/index` cover the architecture behind the
  methods you can run today.
- :doc:`/community/index` lists the channels to use if a method on
  this page does not run on your hardware.
- Browse the source on GitHub at `NVIDIA/flashdreams
  <https://github.com/NVIDIA/flashdreams>`__.
