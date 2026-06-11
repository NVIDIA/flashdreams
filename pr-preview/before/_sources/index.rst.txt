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

            A high-performance inference and serving library for
            interactive autoregressive video and world models,
            and a general platform for real-time world-model applications
            across gaming, autonomous vehicles, robotics, simulated
            or virtual environments, and more!

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

         .. raw:: html

            <div class="fd-promo-video-wrap">
              <video
                class="fd-promo-video-player"
                controls
                playsinline
                preload="metadata"
                aria-label="FlashDreams quick intro video">
                <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/promo_video/flashdreams-promo-hq-6-720P.mp4" type="video/mp4">
              </video>
              <button class="fd-promo-play" type="button" aria-label="Play FlashDreams quick intro video">
                <span class="fd-promo-play-icon" aria-hidden="true"></span>
              </button>
            </div>
            <script>
              (() => {
                const script = document.currentScript;
                const container = script ? script.previousElementSibling : null;
                if (!container) {
                  return;
                }
                const video = container.querySelector(".fd-promo-video-player");
                const playButton = container.querySelector(".fd-promo-play");
                if (!video || !playButton) {
                  return;
                }
                const showOverlay = () => container.classList.remove("is-playing");
                const hideOverlay = () => container.classList.add("is-playing");
                playButton.addEventListener("click", () => {
                  video.controls = true;
                  const playPromise = video.play();
                  if (playPromise && typeof playPromise.catch === "function") {
                    playPromise.catch(showOverlay);
                  }
                });
                video.addEventListener("play", hideOverlay);
                video.addEventListener("pause", showOverlay);
                video.addEventListener("ended", showOverlay);
              })();
            </script>

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

Each tile shows the steady-state per-step speedup — post-warmup,
post-graph-capture — over a separate existing implementation of the
same model. Both runs use the same weights on the same GPU, so the
gain comes from FlashDreams' runtime alone; each tile names its
baseline below. Full methodology lives on the
:doc:`Models page </models/index>`.

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

      .. container:: fd-stat-note

         GB300, vs FastVideo baseline (362 ms → 171 ms per step).

   .. grid-item-card::
      :link: models/lingbot_world.html#profiling-benchmark
      :link-type: url
      :class-card: fd-stat

      .. container:: fd-stat-value

         3.10×

      .. container:: fd-stat-label

         LingBot-World speedup

      .. container:: fd-stat-note

         H100 (4×GPU), vs Official baseline (1950 ms → 629 ms per step).

   .. grid-item-card::
      :link: models/wan21.html#profiling-benchmark
      :link-type: url
      :class-card: fd-stat

      .. container:: fd-stat-value

         1.40×

      .. container:: fd-stat-label

         Wan2.1 speedup

      .. container:: fd-stat-note

         GB300, vs FastVideo baseline (534 ms → 382 ms per step).

   .. grid-item-card::
      :link: models/index.html
      :link-type: url
      :class-card: fd-stat

      .. container:: fd-stat-value

         8

      .. container:: fd-stat-label

         Integrated models

      .. container:: fd-stat-note

         Streaming and bidirectional recipes, one CLI.

Try FlashDreams!
----------------

FlashDreams brings best-in-class per-step latency to interactive
autoregressive video and world models: multiple integrated models across
streaming and bidirectional recipes, multi-GPU execution, and one CLI
to drive them all.

The :doc:`Get Started guide <quickstart/index>` walks from a fresh
checkout to a generated frame on a single GPU.

.. container:: fd-cta-row

   .. button-ref:: quickstart/index
      :ref-type: doc
      :color: primary

      Get Started!

Supported Models
----------------

Streaming and autoregressive recipes emit per-step output with
sub-second latency once warm; bidirectional recipes are kept as
full-block parity references. Each model page carries the canonical
invocation, the checkpoint source, and the per-recipe knobs.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Self-Forcing
      :class-card: fd-feature
      :link: models/self_forcing
      :link-type: doc

      Streaming Wan 2.1 T2V via the Self-Forcing plugin. Sub-second
      steps after warmup on H100 / GB200.

   .. grid-item-card:: Causal-Forcing
      :class-card: fd-feature
      :link: models/causal_forcing
      :link-type: doc

      Causal-forcing framewise T2V and I2V variants of Wan 2.1 via
      the Causal-Forcing plugin.

   .. grid-item-card:: Causal Wan 2.2
      :class-card: fd-feature
      :link: models/causal_wan22
      :link-type: doc

      FastVideo Wan 2.2 14B causal T2V recipe.

   .. grid-item-card:: LingBot-World
      :class-card: fd-feature
      :link: models/lingbot_world
      :link-type: doc

      Camera-controlled I2V with bundled prompt, first-frame, and
      camera arrays.

   .. grid-item-card:: OmniDreams
      :class-card: fd-feature
      :link: models/omnidreams
      :link-type: doc

      Single-view and multi-view streaming recipes against the
      OmniDreams checkpoints, including a diffusion-forcing AR
      variant.

   .. grid-item-card:: FlashVSR
      :class-card: fd-feature
      :link: models/flashvsr
      :link-type: doc

      Streaming video super-resolution for the FlashVSR checkpoint
      family.

   .. grid-item-card:: Wan 2.1 (bidirectional)
      :class-card: fd-feature
      :link: models/wan21
      :link-type: doc

      Bidirectional reference model used for parity testing — T2V
      1.3B / 480p and I2V 14B / 480p.

   .. grid-item-card:: Cosmos-Predict2.5 (bidirectional)
      :class-card: fd-feature
      :link: models/cosmos_predict2
      :link-type: doc

      Bidirectional Cosmos-Predict2 reference recipes (T2V / I2V, 2B).

.. Master toctree: one flat entry per top-level navbar item. Order
   here = order in the navbar.

.. toctree::
   :hidden:
   :maxdepth: 1

   Get Started <quickstart/index>
   Documentation <documentation>
   models/index
   Contribute <community/index>
