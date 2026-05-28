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

.. Hidden H1: the visual title lives in the hero band below, but Sphinx
   still needs a document title for the toctree caption and browser tab.

FlashDreams
===========

.. container:: fd-hero fd-hero-band

   .. container:: fd-split fd-split-asymmetric-reverse

      .. container:: fd-split-text

         .. container:: fd-hero-eyebrow

            Streaming diffusion video

         .. rubric:: Sub-second video diffusion. On every GPU you ship.
            :class: fd-hero-title

         .. container:: fd-hero-lede

            NVIDIA's streaming inference stack for interactive
            autoregressive video and world models. Sub-second
            autoregressive steps after warmup, eight integrated model
            recipes, real-time applications across gaming, autonomous
            vehicles, robotics, and virtual environments — the runtime
            backbone of the `OmniDreams closed-loop demo at GTC 2026
            <https://research.nvidia.com/labs/sil/projects/omnidreams-blog/>`_.

         .. container:: fd-cta-row

            .. button-ref:: quickstart/index
               :ref-type: doc
               :color: primary

               Get started

            .. button-ref:: benchmarks/index
               :ref-type: doc
               :color: secondary
               :outline:

               See the benchmarks

            .. button-link:: https://github.com/NVIDIA/flashdreams
               :color: secondary
               :outline:

               Star on GitHub

      .. container:: fd-split-visual

         .. admonition:: PLACEHOLDER — hero illustration / loop
            :class: placeholder

            **What goes here:** A looping 5–10s clip of one of the streaming
            recipes (e.g. ``self-forcing-wan2.1-t2v-1.3b-taehv``) running
            end-to-end, or a stylised architecture illustration of the
            KV cache + ring attention + CUDA-graph pipeline.

            **Format:** ``_static/hero-loop.avif`` (animated AVIF, 16:9,
            autoplay loop) or a static ``_static/hero-illustration.svg``.

Headline numbers
----------------

.. container:: fd-eyebrow

   Steady-state, post-warmup

.. container:: fd-lede

   Per-step latency on the same recipe, same GPU, against the upstream
   library's own runner. Each tile names the GPU and the specific
   baseline; full methodology lives on the :doc:`benchmarks page
   <benchmarks/index>`.

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

            Streaming and bidirectional recipes, one command-line front door.

Born from OmniDreams
--------------------

.. container:: fd-eyebrow

   Origin story

.. container:: fd-split fd-split-asymmetric-reverse

   .. container:: fd-split-text

      FlashDreams began as the optimized runtime behind the `OmniDreams
      closed-loop demo for GTC 2026
      <https://research.nvidia.com/labs/sil/projects/omnidreams-blog/>`_,
      and has grown into a general platform for real-time world-model
      applications — driving simulators, autonomous-vehicle scene
      rollout, embodied robotics, gaming, and virtual environments.

      The same inference pipeline that ships in OmniDreams now hosts
      seven other model families, and the same per-AR-step latency
      targets carry across all of them.

   .. container:: fd-split-visual

      .. admonition:: PLACEHOLDER — OmniDreams hero loop
         :class: placeholder

         **What goes here:** A short looping clip from the OmniDreams
         multi-view streaming recipe — closed-loop scene rollout,
         multiple simultaneous camera views, generated end-to-end by
         FlashDreams. ``_static/omnidreams-loop.avif`` (animated AVIF,
         16:9). Until that lands, the :doc:`OmniDreams model page
         <models/omnidreams>` carries the canonical sample videos.

Why FlashDreams
---------------

.. container:: fd-eyebrow

   Designed for streaming, not retrofitted

.. container:: fd-split fd-split-asymmetric

   .. container:: fd-split-text

      Diffusion video generators are getting fast enough to be
      interactive — but only if the inference stack is built for it.
      FlashDreams is organised around three sharp abstractions that every
      shipped recipe plugs into.

      **KV-cached transformers.** Each autoregressive chunk re-uses prior
      context as a KV cache instead of recomputing it. Self-forcing and
      causal-forcing training regimes are first-class.

      **Ring attention.** Context-parallel attention across ranks.
      Long-horizon generation scales out instead of OOM-ing on a
      single GPU.

      **CUDA-graph capture.** The steady-state forward is captured into a
      CUDA graph after warmup, collapsing Python and launch overhead in
      the hot loop.

   .. container:: fd-split-visual

      .. admonition:: PLACEHOLDER — architecture diagram
         :class: placeholder

         Stacked-block diagram showing the three abstractions in the
         hot loop: KV cache → ring attention → CUDA-graph capture.
         Saved as ``_static/arch-diagram.svg`` (light and dark
         variants).

The result is the per-step latency in the :doc:`benchmarks
<benchmarks/index>` table — sub-second per autoregressive chunk on H100
/ GB200 once the pipeline is warm.

See it in motion
----------------

.. container:: fd-eyebrow

   Live demo loops

.. container:: fd-lede

   Streaming recipes generating side-by-side against their upstream
   references. Each loop is < 10s, captured at native resolution.

.. container:: fd-media-rail

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — Self-Forcing T2V loop
         :class: placeholder

         5–10s looping clip of ``self-forcing-wan2.1-t2v-1.3b-taehv``
         generating from a held-out prompt. Side-by-side with the
         FastVideo baseline so the wall-clock difference reads
         visually.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — Causal-Forcing I2V loop
         :class: placeholder

         5–10s looping clip of the causal-forcing I2V recipe driving
         a Wan 2.1 14B checkpoint. Multi-frame autoregressive output
         from a single first-frame condition.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — OmniDreams multi-view
         :class: placeholder

         Four simultaneous view frames generated by the OmniDreams
         multi-view recipe. Same scene, four camera angles, single
         streaming pipeline.

What's inside
-------------

.. container:: fd-eyebrow

   Library surface

.. container:: fd-lede

   A core library, an infra layer, a plugin-friendly recipe registry,
   and a single command-line front door for every shipped model.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: One CLI for every model
      :class-card: fd-feature
      :columns: 12 12 6 6
      :link: api/cli
      :link-type: doc

      One console script dispatches every shipped recipe. Each model
      is a named slug; every overridable field is a flag. New recipes
      auto-register through entry points — no central manifest, no
      per-model command to remember.

   .. grid-item-card:: Multi-GPU by default
      :class-card: fd-feature
      :columns: 12 12 6 6
      :link: api/infra
      :link-type: doc

      Recipes auto-detect CP size from ``torch.distributed``'s world
      group. The launcher is the single source of truth.

   .. grid-item-card:: Plugin-friendly recipes
      :class-card: fd-feature
      :columns: 12 12 6 6
      :link: developer_guides/new_integration
      :link-type: doc

      Recipes can ship in-tree or as out-of-tree plugins, discovered
      through entry points. Self-Forcing and Causal-Forcing ship as
      workspace-member packages under ``integrations/``.

   .. grid-item-card:: Mockable for CI
      :class-card: fd-feature
      :columns: 12 12 6 6
      :link: developer_guides/inference_pipeline_overview
      :link-type: doc

      Heaviest C-extensions (transformer-engine, triton, pynvml) are
      mocked at autodoc time so docs build on CPU-only hosts.

Supported models
----------------

.. container:: fd-eyebrow

   Eight recipes, one CLI

.. container:: fd-lede

   Streaming and autoregressive variants emit per-AR-step output with
   sub-second latency once warm; bidirectional variants are the
   single full-block reference used for parity. Each model page has
   the canonical invocation, the checkpoint source, and the per-recipe
   knobs.

.. container:: fd-media-rail fd-media-rail-4

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — Self-Forcing sample frame
         :class: placeholder

         Still from a representative Self-Forcing T2V run.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — Causal-Forcing sample frame
         :class: placeholder

         Still from a Causal-Forcing I2V run.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — LingBot-World sample frame
         :class: placeholder

         Still from a LingBot-World camera-controlled I2V run.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — OmniDreams multi-view still
         :class: placeholder

         Multi-view OmniDreams composite frame.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Self-Forcing
      :class-card: fd-feature
      :link: models/self_forcing
      :link-type: doc

      Streaming Wan 2.1 T2V via the Self-Forcing plugin. AR steps
      after warmup are sub-second on H100 / GB200.

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

      Camera-controlled I2V with bundled prompt + first-frame +
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

      Streaming video super-resolution for the FlashVSR
      checkpoint family.

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

How it compares
---------------

.. container:: fd-eyebrow

   Same recipe, same hardware

.. container:: fd-lede

   FlashDreams runs the same model code as the library it integrates
   with — measured in wall-clock at steady state, post-warmup,
   post-graph-capture, post-JIT.

.. container:: fd-split fd-split-reverse

   .. container:: fd-split-text

      We rebuild the upstream runner on identical hardware, then run
      FlashDreams against the same checkpoint and sampling
      configuration. The delta is what the per-step latency table on
      the benchmarks page shows; numbers update with the rolling
      ``main`` branch whenever a recipe lands a perf change.

      Full methodology — what gets timed, what we drop as warmup,
      how the upstream baseline is built — lives on the
      :doc:`benchmarks page <benchmarks/index>`.

   .. container:: fd-split-visual

      .. admonition:: PLACEHOLDER — comparison still
         :class: placeholder

         Side-by-side still frame: FlashDreams output (left) vs
         upstream baseline output (right), same prompt, same seed.
         Overlaid wall-clock readout in the bottom-right corner.

Quick start
-----------

.. container:: fd-eyebrow

   Two commands to a generated video

.. container:: fd-split fd-split-asymmetric

   .. container:: fd-split-text

      Sync the workspace, list the shipped recipes, run one. The first
      sync compiles CUDA extensions for your local GPU; once that's
      done, every recipe is a single command away.

      The :doc:`quickstart/index` is the annotated walkthrough — fresh
      checkout to a frame on disk on a single GPU. From there the
      :doc:`developer guides </developer_guides/index>` cover the
      pipeline internals, distributed launching, and how to ship a
      custom recipe of your own.

      .. admonition:: New to streaming diffusion?
         :class: fd-callout

         Read the :doc:`inference pipeline overview
         </developer_guides/inference_pipeline_overview>` before
         tweaking recipes — it walks the hot loop end to end.

   .. container:: fd-split-visual

      .. code-block:: bash

         uv sync --extra dev --extra runners
         uv run flashdreams-run --help

         # Single-GPU streaming inference
         # (Self-Forcing Wan 2.1 T2V).
         uv run flashdreams-run \
             self-forcing-wan2.1-t2v-1.3b-taehv \
             --total-blocks 7

.. rst-class:: fd-band-accent fd-cta-banner

Try FlashDreams
---------------

.. container:: fd-eyebrow

   Apache-2.0, developed in the open

.. container:: fd-lede

   Star the repo, run the quickstart, file an issue when a number
   doesn't reproduce on your hardware. FlashDreams is built in the
   open at `NVIDIA/flashdreams
   <https://github.com/NVIDIA/flashdreams>`_.

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

.. Master toctree: one flat entry per top-level navbar item. Order
   here = order in the navbar. Each section's own index page owns the
   toctree to its sub-pages (which drives the per-section sidebar).

.. toctree::
   :hidden:
   :maxdepth: 1

   benchmarks/index
   Getting Started <quickstart/index>
   developer_guides/index
   models/index
   CLI/API References <api/index>
   community/index
