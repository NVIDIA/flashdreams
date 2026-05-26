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

.. Hidden H1: the visual title lives in the hero below, but Sphinx still
   needs a document title for the toctree caption and the browser tab.

FlashDreams
===========

.. container:: fd-hero

   .. container:: fd-hero-eyebrow

      Streaming diffusion video

   .. rubric:: FlashDreams
      :class: fd-hero-title

   .. container:: fd-hero-lede

      Sub-second autoregressive video diffusion on a single GPU.
      FlashDreams is the streaming inference stack for diffusion-based
      video — KV-cached transformers, ring attention, and CUDA-graph
      capture, behind one ``flashdreams-run`` CLI.

   .. container:: fd-cta-row

      .. button-ref:: getting_started/index
         :ref-type: doc
         :color: primary

         Get started

      .. button-ref:: apis/index
         :ref-type: doc
         :color: secondary
         :outline:

         API Reference

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            TBD

         .. container:: fd-stat-label

            Steady-state step (ms)

         .. container:: fd-stat-note

            H100, ``self-forcing-wan2.1-t2v-1.3b-flash``.

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            TBD

         .. container:: fd-stat-label

            FlashDreams / upstream

         .. container:: fd-stat-note

            Same recipe, same GPU; ratio > 1 means faster.

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            TBD

         .. container:: fd-stat-label

            Peak memory (GiB)

         .. container:: fd-stat-note

            Steady-state, post-graph-capture.

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            TBD

         .. container:: fd-stat-label

            Scaling at 8 GPUs

         .. container:: fd-stat-note

            Ring attention, bidirectional reference.

.. admonition:: PLACEHOLDER — headline stat values
   :class: placeholder

   **What goes here:** four ≤ 6-character stat values for the grid
   above, mirroring the canonical grid on :doc:`benchmarks/index`.

   **Source data:** rows of the Results tables on the benchmarks page;
   do not invent.

   **Reproduce with:** the per-row CLI invocations in *Methodology* on
   the benchmarks page.

----

Showcase
--------

.. admonition:: PLACEHOLDER — showcase video
   :class: placeholder

   **What goes here:** A 10–30s reel showing FlashDreams generating video
   end-to-end — ideally a side-by-side of one of the streaming recipes
   (e.g. ``self-forcing-wan2.1-t2v-1.3b-flash``) against its upstream
   reference, plus a frame of the multi-view OmniDreams output.

   **Format:** ``_static/showcase.mp4`` (self-hosted, 16:9, < 10 MB) or a
   YouTube-nocookie embed ID.

   **Source / coordinate with:** marketing + the recipe maintainers
   (sample prompts already live in each ``examples/*.rst`` walkthrough).

----

Why FlashDreams
---------------

Diffusion video generators are getting fast enough to be interactive, but
only if the inference stack is built for it. FlashDreams is organised
around a few sharp abstractions — documented in the
:doc:`apis/index` — that every recipe
plugs into: **KV-cached transformers** so each autoregressive chunk
re-uses prior context instead of recomputing it, **ring attention**
across context-parallel ranks for long sequences without OOM, and
**CUDA-graph capture** of the steady-state forward so per-step overhead
collapses once the pipeline is warm. The result is the streaming-step
latencies recorded in the `PERFORMANCE.md
<https://github.com/NVIDIA/flashdreams/blob/main/PERFORMANCE.md>`_
profiles — sub-second per autoregressive chunk on H100 / GB200 after
warmup for the Self-Forcing Wan 2.1 T2V recipe.

A single ``flashdreams-run`` CLI fronts every shipped recipe, in-tree or
plugin, with every overridable field exposed as a flag. The same command
that does a single-GPU debug run scales to multi-GPU context-parallelism
under ``torchrun``; recipes auto-detect their CP size from the world
group, so there is no separate distributed-launcher config to maintain.
The :doc:`getting_started/index` has the
annotated walkthrough.

----

What's inside
-------------

.. grid:: 1 2 2 3
   :gutter: 3
   :margin: 0 0 4 0

   .. grid-item-card:: KV-cached transformers
      :class-card: fd-feature
      :link: apis/infra
      :link-type: doc

      First-class support for autoregressive flow-matching models with
      self-forcing and causal-forcing training regimes; prior chunks
      stay resident as KV so each AR step only attends to fresh latents.

   .. grid-item-card:: Ring attention
      :class-card: fd-feature
      :link: apis/core
      :link-type: doc

      Context-parallel attention across ranks, so long-horizon
      generation scales out instead of OOM-ing on a single GPU.

   .. grid-item-card:: CUDA-graph capture
      :class-card: fd-feature
      :link: apis/recipes
      :link-type: doc

      The steady-state forward is captured into a CUDA graph after warmup,
      eliminating Python and launch overhead from the per-step hot path.

   .. grid-item-card:: Unified ``flashdreams-run`` CLI
      :class-card: fd-feature
      :link: reference/cli
      :link-type: doc

      One console script dispatches over every in-tree and plugin-provided
      recipe. Each overridable field is a CLI flag; ``--help`` lists every
      registered runner.

   .. grid-item-card:: Multi-GPU by default
      :class-card: fd-feature
      :link: apis/infra
      :link-type: doc

      Recipes auto-detect CP size from ``torch.distributed``'s world
      group. The launcher is the single source of truth — no ``cp_size``
      knob to drift out of sync.

   .. grid-item-card:: Plugin-friendly recipes
      :class-card: fd-feature
      :link: developer_guides/new_recipes
      :link-type: doc

      Recipes can ship in-tree or as out-of-tree plugins discovered
      through entry points. Self-Forcing and Causal-Forcing ship as
      workspace-member packages under ``integrations/``.

----

Supported models
----------------

Each model below has a walkthrough with the exact ``flashdreams-run``
slug, the checkpoint source, and any per-recipe knobs. Recipes are split
into **streaming / autoregressive** (KV-cached, per-AR-step output) and
**bidirectional** (single full-block reference) variants.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Self-Forcing
      :class-card: fd-feature
      :link: models/self_forcing
      :link-type: doc

      Streaming Wan 2.1 T2V via the Self-Forcing plugin. AR steps after
      warmup are sub-second on H100 / GB200.

   .. grid-item-card:: Causal-Forcing
      :class-card: fd-feature
      :link: models/causal_forcing
      :link-type: doc

      Causal-forcing framewise T2V and I2V variants of Wan 2.1 via the
      Causal-Forcing plugin.

   .. grid-item-card:: Causal Wan 2.2
      :class-card: fd-feature
      :link: models/fastvideo_wan22
      :link-type: doc

      FastVideo Wan 2.2 14B causal T2V recipe.

   .. grid-item-card:: Lingbot-World
      :class-card: fd-feature
      :link: models/lingbot_world
      :link-type: doc

      Camera-controlled I2V with bundled prompt + first-frame + camera
      arrays.

   .. grid-item-card:: OmniDreams
      :class-card: fd-feature
      :link: models/omnidreams
      :link-type: doc

      Single-view and multi-view streaming recipes against the OmniDreams
      checkpoints, including a diffusion-forcing AR variant.

   .. grid-item-card:: Wan 2.1 (bidirectional)
      :class-card: fd-feature
      :link: models/wan21
      :link-type: doc

      Single bidirectional reference model used for parity testing —
      T2V 1.3B / 480p and I2V 14B / 480p.

----

Quick start
-----------

FlashDreams is a `uv <https://docs.astral.sh/uv/>`_ workspace. The CLI
lazy-imports ``mediapy`` + ``opencv`` for I/O, so install the ``runners``
extra whenever you want to actually generate videos:

.. code-block:: bash

   uv sync --extra dev --extra runners
   uv run flashdreams-run --help

   # Single-GPU streaming inference (Self-Forcing Wan 2.1 T2V).
   uv run flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b-flash --total-blocks 7

.. admonition:: New to streaming diffusion?
   :class: fd-callout

   The :doc:`getting_started/index` has the
   annotated quickstart, an end-to-end first-generation walkthrough, and
   pointers into the developer guides for CUDA-graph capture, distributed
   launching, and authoring a custom recipe.

----

Join the community
------------------

FlashDreams is developed in the open at
`NVIDIA/flashdreams <https://github.com/NVIDIA/flashdreams>`_.

.. container:: fd-cta-row

   .. button-link:: https://github.com/NVIDIA/flashdreams
      :color: primary

      GitHub

   .. button-ref:: community/index
      :ref-type: doc
      :color: secondary
      :outline:

      Community

   .. button-ref:: community/contributing
      :ref-type: doc
      :color: secondary
      :outline:

      Contributing

   .. button-ref:: benchmarks/index
      :ref-type: doc
      :color: secondary
      :outline:

      Benchmarks

.. Master toctree: one flat entry per top-level navbar item. Order
   here = order in the navbar. Each section's own index page owns the
   toctree to its sub-pages (which drives the per-section sidebar).

.. toctree::
   :hidden:
   :maxdepth: 1

   benchmarks/index
   getting_started/index
   developer_guides/index
   models/index
   CLI Reference <reference/index>
   API Reference <apis/index>
   community/index
