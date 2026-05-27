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

Developer Guides
================

.. container:: fd-hero fd-hero-band

   .. container:: fd-hero-eyebrow

      How the system works

   .. rubric:: Beyond the CLI: how FlashDreams runs.
      :class: fd-hero-title

   .. container:: fd-hero-lede

      The quickstart shows how to drive a recipe. These guides cover
      what's happening underneath: the inference pipeline a recipe
      runs through, the configuration layer every recipe shares, and
      the integration surface you plug a new model into.

   .. container:: fd-cta-row

      .. button-ref:: inference_pipeline_overview
         :ref-type: doc
         :color: primary

         Start with the pipeline

      .. button-ref:: /api/index
         :ref-type: doc
         :color: secondary
         :outline:

         Browse the API

Guides
------

.. container:: fd-eyebrow

   Five walkthroughs

.. container:: fd-lede

   Read in order: the pipeline overview anchors the rest. The config
   system is the layer every recipe shares; new integrations sit on
   top of both. Usage patterns and interactive serving cover how to
   embed the pipeline in your own code and keep a streaming session
   alive.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Inference pipeline overview
      :class-card: fd-feature fd-feature-hero
      :columns: 12 12 6 8
      :link: inference_pipeline_overview
      :link-type: doc

      The hot loop a recipe goes through end-to-end: warmup, CUDA-
      graph capture, the AR-step body, the ring-attention shard
      group, finalize. The mental model the rest of the project
      assumes you have.

   .. grid-item-card:: Config system
      :class-card: fd-feature
      :columns: 12 12 6 4
      :link: config_system
      :link-type: doc

      How every overridable field is surfaced as a CLI flag, how
      recipe defaults compose, how to layer overrides on top.

   .. grid-item-card:: Add a new method
      :class-card: fd-feature
      :columns: 12 12 6 4
      :link: new_integration
      :link-type: doc

      The entry-point surface a new recipe ships against — what to
      subclass, what to register, where the parity tests live.

   .. grid-item-card:: Usage patterns
      :class-card: fd-feature
      :columns: 12 12 6 4
      :link: usage_patterns
      :link-type: doc

      Common ways to drive FlashDreams from your own Python — the
      CLI, the in-process runner API, and the pipeline-level surface
      for embedding.

   .. grid-item-card:: Interactive serving
      :class-card: fd-feature
      :columns: 12 12 6 4
      :link: interactive_serving
      :link-type: doc

      Patterns for keeping a streaming session alive: warmup,
      steady-state generation, and how the WebRTC / gRPC servers
      under ``integrations/`` wire the pipeline up.


Where these guides fit
----------------------

.. container:: fd-eyebrow

   Map to the rest of the site

.. container:: fd-split fd-split-reverse

   .. container:: fd-split-text

      The guides on this page are **conceptual** — they describe how
      the system is structured and why. The :doc:`/api/index` is the
      **reference** — every public class, function, and CLI flag
      with its docstring.

      Working forward from a recipe: start with the pipeline
      overview, then read the recipe's per-model page under
      :doc:`/models/index`, then drop into the matching ``api/*``
      module for the implementation details.

   .. container:: fd-split-visual

      .. container:: fd-info-card

         .. container:: fd-info-card-title

            Quickstart

         | Two commands to a generated clip.
         | → :doc:`/quickstart/index`

         .. container:: fd-info-card-title

            Models

         | Per-recipe pages: slugs, checkpoints,
         | per-recipe knobs.
         | → :doc:`/models/index`

         .. container:: fd-info-card-title

            API reference

         | Autodoc surface, organised by module.
         | → :doc:`/api/index`

.. toctree::
   :hidden:
   :maxdepth: 1

   inference_pipeline_overview
   config_system
   new_integration
   usage_patterns
   interactive_serving
