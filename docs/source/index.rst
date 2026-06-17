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

FlashDreams
===========

.. container:: fd-hero fd-hero-band

   .. container:: fd-split fd-split-asymmetric

      .. container:: fd-split-text

         .. rubric:: FlashDreams
            :class: fd-hero-title

         .. container:: fd-hero-lede

            FlashDreams is an inference and serving runtime for turning
            autoregressive video and world models into live, controllable
            simulations. It runs the model in a continuous loop, carrying
            state forward and streaming frames while new actions or sensor
            inputs change what happens next, whether the application is a
            game world, an autonomous-vehicle simulator, robotic policy
            testing, or a virtual training environment.

         .. container:: fd-cta-row

            .. button-ref:: quickstart/index
               :ref-type: doc
               :color: primary

               Get Started!

            .. button-link:: https://github.com/NVIDIA/flashdreams
               :color: secondary
               :outline:

               GitHub

            .. button-ref:: community/index
               :ref-type: doc
               :color: secondary
               :outline:

               Contribute

      .. container:: fd-split-visual

         .. container:: fd-promo-video-wrap

            .. image:: /_static/promo/flashdreams-promo.avif
               :alt: FlashDreams quick intro animation
               :class: fd-promo-video-player zoomable

Why FlashDreams?
----------------

.. container:: fd-split fd-split-reverse fd-split-asymmetric-reverse

   .. container:: fd-split-text

      A world model learns to generate and evolve an environment over time. In
      practice that usually means video, but the same idea extends to actions,
      state, audio, sensor input, and control signals. Serving one means keeping
      a session alive while input, model state, GPU inference, and output advance
      together, rather than producing a single static clip, which is what makes
      interactive simulation, robotics, autonomy, and game-like experiences
      possible.

   .. container:: fd-split-visual

      .. image:: /_static/diagrams/compare-offline-online-video-model-v2.jpg
         :alt: Offline one-shot video inference compared with online autoregressive world-model serving.
         :class: zoomable

FlashDreams is built for that real-time case: a closed-loop world-model
demo, a driving simulator, an interactive scene rollout. Generating
high-quality video is not enough on its own. The runtime has to keep an
interactive session responsive while the model continues to advance the
world. That comes down to four things:

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: Low latency
      :class-card: fd-feature

      Keep the interaction responsive when controls, sensors, or user
      input change.

   .. grid-item-card:: High throughput
      :class-card: fd-feature

      Keep the GPU busy across autoregressive steps and multi-GPU
      execution.

   .. grid-item-card:: Steady streaming generation
      :class-card: fd-feature

      Stream frames or chunks at a steady pace while the session
      continues.

   .. grid-item-card:: World-state evolution
      :class-card: fd-feature

      Carry rolling state forward so the generated world evolves across
      steps.

Performance
-----------

Each tile shows the speedup over a separate existing implementation of
the same model. Both runs use the same weights on the same GPU, so the
gain comes from FlashDreams' runtime alone. Each tile links to the
profiling chart on its model page.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card::
      :link: models/self_forcing.html#profiling-benchmark
      :link-type: url
      :class-card: fd-stat

      .. container:: fd-stat-value

         2.12×

      .. container:: fd-stat-label

         Self-Forcing speedup

   .. grid-item-card::
      :link: models/lingbot_world.html#profiling-benchmark
      :link-type: url
      :class-card: fd-stat

      .. container:: fd-stat-value

         3.10×

      .. container:: fd-stat-label

         LingBot-World speedup

   .. grid-item-card::
      :link: models/wan21.html#profiling-benchmark
      :link-type: url
      :class-card: fd-stat

      .. container:: fd-stat-value

         1.40×

      .. container:: fd-stat-label

         Wan2.1 speedup

   .. grid-item-card::
      :link: models/flashvsr.html#profiling-benchmark
      :link-type: url
      :class-card: fd-stat

      .. container:: fd-stat-value

         1.42×

      .. container:: fd-stat-label

         FlashVSR speedup

Try FlashDreams!
----------------

FlashDreams brings best-in-class per-step latency to interactive
autoregressive video and world models: multiple integrated models across
streaming and bidirectional methods, multi-GPU execution, and one CLI
to drive them all.

The :doc:`Get Started guide <quickstart/index>` walks from a fresh
checkout to running OmniDreams, an interactive driving world-model demo
built on FlashDreams.

Supported Models
----------------

Streaming and autoregressive model implementations emit per-step output with
sub-second latency once warm; bidirectional model implementations are kept as
full-block parity references. Each model page carries the canonical
invocation, the checkpoint source, and the per-implementation knobs.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: OmniDreams
      :class-card: fd-feature
      :link: models/omnidreams
      :link-type: doc

      Interactive world simulator for autonomous vehicles.

   .. grid-item-card:: Self-Forcing
      :class-card: fd-feature
      :link: models/self_forcing
      :link-type: doc

      Autoregressive text-to-video based on Wan 2.1.

   .. grid-item-card:: Causal-Forcing
      :class-card: fd-feature
      :link: models/causal_forcing
      :link-type: doc

      Autoregressive text/image-to-video based on Wan 2.1.

   .. grid-item-card:: Causal Wan 2.2
      :class-card: fd-feature
      :link: models/causal_wan22
      :link-type: doc

      Autoregressive text-to-video based on Wan 2.2 from FastVideo.

   .. grid-item-card:: LingBot-World
      :class-card: fd-feature
      :link: models/lingbot_world
      :link-type: doc

      Camera-controllable image-to-video world model.

   .. grid-item-card:: FlashVSR
      :class-card: fd-feature
      :link: models/flashvsr
      :link-type: doc

      Streaming video super-resolution.

   .. grid-item-card:: Wan 2.1 (bidirectional)
      :class-card: fd-feature
      :link: models/wan21
      :link-type: doc

      Bidirectional video generation model that supports both
      text-to-video and image-to-video.

   .. grid-item-card:: Cosmos-Predict2.5 (bidirectional)
      :class-card: fd-feature
      :link: models/cosmos_predict2
      :link-type: doc

      Bidirectional Cosmos-Predict2 reference implementations (T2V / I2V, 2B).

.. Master toctree: one flat entry per top-level navbar item. Order
   here = order in the navbar.

.. toctree::
   :hidden:
   :maxdepth: 1

   Get Started <quickstart/index>
   Documentation <documentation>
   models/index
   Contribute <community/index>
