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

Getting Started
===============

.. container:: fd-hero fd-hero-band

   .. container:: fd-hero-eyebrow

      Two commands to a generated video

   .. rubric:: From clone to first frame in five minutes.
      :class: fd-hero-title

   .. container:: fd-hero-lede

      The two walkthroughs below take a fresh checkout to a generated
      clip on a single GPU. Install once, pick a streaming recipe, watch
      the per-step timings drop below a second as the pipeline warms.

   .. container:: fd-cta-row

      .. button-ref:: installation
         :ref-type: doc
         :color: primary

         Install FlashDreams

      .. button-ref:: first_world_model
         :ref-type: doc
         :color: secondary
         :outline:

         Run your first model

What you'll do
--------------

.. container:: fd-eyebrow

   The shortest path through

.. container:: fd-split fd-split-asymmetric

   .. container:: fd-split-text

      The two pages below are the canonical walkthrough. They assume
      a fresh machine with a CUDA-capable GPU and walk through the
      bare minimum to reach a generated clip on disk. Once those are
      behind you, the :doc:`developer guides </developer_guides/index>`
      cover the system architecture: CUDA-graph capture, ring
      attention, distributed launching, custom recipes.

      One command-line front door is the only entry point you need.
      Every shipped recipe is a named slug; every overridable field
      is a flag.

   .. container:: fd-split-visual

      .. code-block:: bash

         # 1. Install
         uv sync --extra dev --extra runners

         # 2. List shipped recipes
         uv run flashdreams-run --help

         # 3. Run a streaming recipe
         uv run flashdreams-run \
             self-forcing-wan2.1-t2v-1.3b-taehv \
             --total-blocks 7

Walkthroughs
------------

.. container:: fd-eyebrow

   Two short reads, ordered

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Install FlashDreams
      :class-card: fd-feature fd-feature-hero
      :columns: 12 12 6 6
      :link: installation
      :link-type: doc

      The minimum-friction install path: a pinned PyTorch + CUDA
      toolchain, plus the optional I/O extras needed to write generated
      video to disk.

   .. grid-item-card:: Launch your first model
      :class-card: fd-feature
      :columns: 12 12 6 6
      :link: first_world_model
      :link-type: doc

      Pick a streaming recipe, run it, read the per-step log to watch
      the latency settle as the CUDA graph captures and the pipeline
      reaches steady state.

After the walkthroughs
----------------------

.. container:: fd-eyebrow

   Where to go next

.. container:: fd-lede

   The quickstart pages get you to a frame on disk. Past that the
   site is organised by what you'll want to do next.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Pick a different recipe
      :class-card: fd-feature
      :link: /models/index
      :link-type: doc

      Per-model walkthroughs with the CLI slug, checkpoint source,
      and per-recipe knobs.

   .. grid-item-card:: Understand the pipeline
      :class-card: fd-feature
      :link: /developer_guides/inference_pipeline_overview
      :link-type: doc

      The hot loop, end to end: KV cache, ring attention,
      CUDA-graph capture.

   .. grid-item-card:: See the benchmarks
      :class-card: fd-feature
      :link: /benchmarks/index
      :link-type: doc

      Steady-state numbers, parity status, reproducer CLIs.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   first_world_model
