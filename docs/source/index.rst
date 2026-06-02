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

   .. container:: fd-hero-lede

      A high-performance inference and serving library for
      interactive autoregressive video and world models.

   .. container:: fd-cta-row

      .. button-ref:: quickstart/index
         :ref-type: doc
         :color: primary

         Get started

      .. button-link:: https://github.com/NVIDIA/flashdreams
         :color: secondary
         :outline:

         GitHub

      .. button-ref:: community/index
         :ref-type: doc
         :color: secondary
         :outline:

         Community

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

Why FlashDreams
---------------

FlashDreams is built for the case where a diffusion video model has to
respond in real time — a closed-loop world-model demo, a driving
simulator, an interactive scene rollout. The optimisations needed for
that case are different from those used by an offline, one-shot video
generator, and FlashDreams organises them into three abstractions that
every shipped recipe uses.

**KV-cached transformers.** Each autoregressive chunk re-uses prior
context as a KV cache instead of recomputing it. Self-forcing and
causal-forcing training regimes are first-class.

**Ring attention.** Context-parallel attention across ranks, so
long-horizon generation scales out instead of OOM-ing on a single GPU.

**CUDA-graph capture.** The steady-state forward is captured into a
CUDA graph after warmup, collapsing Python and launch overhead in the
hot loop.

The library is Apache-2.0 and developed in the open. The internals are
covered in the :doc:`documentation <documentation>`.

Performance
-----------

Each tile shows per-step latency at steady state — post-warmup,
post-graph-capture — measured against the upstream library's own
runner on the same hardware and the same checkpoint. Full methodology
lives on the :doc:`benchmarks page </models/index>`.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            2.12×

         .. container:: fd-stat-label

            Self-Forcing speedup

         .. container:: fd-stat-note

            GB300, vs FastVideo baseline (362 ms → 171 ms per step).

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            3.10×

         .. container:: fd-stat-label

            LingBot-World speedup

         .. container:: fd-stat-note

            H100, vs Official baseline (1950 ms → 629 ms per step).

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            1.40×

         .. container:: fd-stat-label

            Wan2.1 speedup

         .. container:: fd-stat-note

            GB300, vs FastVideo baseline (534 ms → 382 ms per step).

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            8

         .. container:: fd-stat-label

            Integrated models

         .. container:: fd-stat-note

            Streaming and bidirectional recipes, one CLI.

Try FlashDreams
---------------

The :doc:`Get Started guide </quickstart/index>` walks from a fresh
checkout to a generated frame on a single GPU.

.. container:: fd-cta-row

   .. button-ref:: quickstart/index
      :ref-type: doc
      :color: primary

      Get started

Supported models
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
   Benchmarks <models/index>
   community/index
