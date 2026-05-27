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

.. container:: fd-hero fd-hero-band

   .. container:: fd-hero-eyebrow

      Eight recipes, one CLI

   .. rubric:: Every shipped model. One command line away.
      :class: fd-hero-title

   .. container:: fd-hero-lede

      Streaming and autoregressive recipes (KV-cached, per-AR-step
      output) and bidirectional reference recipes (single full-block
      reference, used for parity). Each model page has the canonical
      invocation, the checkpoint source, and the per-recipe knobs.

   .. container:: fd-cta-row

      .. button-link:: https://github.com/NVIDIA/flashdreams
         :color: primary

         Browse on GitHub

      .. button-ref:: /developer_guides/new_integration
         :ref-type: doc
         :color: secondary
         :outline:

         Add your own model

Running a model
---------------

.. container:: fd-eyebrow

   One front door, every recipe

.. container:: fd-split fd-split-asymmetric

   .. container:: fd-split-text

      Every recipe is a named slug. Streaming runners take a
      ``--total-blocks`` flag to bound the autoregressive step count;
      bidirectional runners produce a single end-to-end output. Every
      overridable field on every recipe is exposed as a CLI flag.

      Each model page below has the canonical invocation for that
      recipe — copy, paste, run.

   .. container:: fd-split-visual

      .. code-block:: bash

         # List every shipped recipe.
         uv run flashdreams-run --help

         # Streaming Wan 2.1 T2V (Self-Forcing plugin).
         uv run flashdreams-run \
             self-forcing-wan2.1-t2v-1.3b-flash \
             --total-blocks 7

         # Camera-controlled I2V with bundled example data.
         uv run flashdreams-run \
             lingbot-world-fast \
             --example-data True --total-blocks 21

Streaming / autoregressive
--------------------------

.. container:: fd-eyebrow

   KV-cached, per-AR-step output

.. container:: fd-lede

   Designed for interactive use — output emits as the pipeline
   advances, with sub-second steady-state step latency on H100 /
   GB200 once the CUDA graph is captured.

.. container:: fd-media-rail

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — Self-Forcing demo frame
         :class: placeholder

         Sample still from a Self-Forcing T2V generation.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — Causal-Forcing demo frame
         :class: placeholder

         Sample still from a Causal-Forcing I2V run.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — LingBot-World demo frame
         :class: placeholder

         Sample still from a LingBot-World camera-controlled I2V run.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Self-Forcing
      :class-card: fd-feature
      :link: self_forcing
      :link-type: doc

      Streaming Wan 2.1 T2V via the Self-Forcing plugin. AR steps
      after warmup are sub-second on H100 / GB200.

   .. grid-item-card:: Causal-Forcing
      :class-card: fd-feature
      :link: causal_forcing
      :link-type: doc

      Causal-forcing framewise T2V and I2V variants of Wan 2.1 via
      the Causal-Forcing plugin.

   .. grid-item-card:: Causal Wan 2.2
      :class-card: fd-feature
      :link: causal_wan22
      :link-type: doc

      FastVideo Wan 2.2 14B causal T2V recipe.

   .. grid-item-card:: LingBot-World
      :class-card: fd-feature
      :link: lingbot_world
      :link-type: doc

      Camera-controlled I2V with bundled prompt + first-frame +
      camera arrays.

   .. grid-item-card:: OmniDreams
      :class-card: fd-feature
      :link: omnidreams
      :link-type: doc

      Single-view and multi-view streaming recipes against the
      OmniDreams checkpoints, including a diffusion-forcing AR
      variant.

   .. grid-item-card:: FlashVSR
      :class-card: fd-feature
      :link: flashvsr
      :link-type: doc

      Streaming video super-resolution recipe for the FlashVSR
      checkpoint family.

Bidirectional reference
-----------------------

.. container:: fd-eyebrow

   Single full-block reference

.. container:: fd-lede

   Used for parity testing against the streaming recipes above.
   These runners emit one end-to-end output per invocation rather
   than per-step.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Wan 2.1
      :class-card: fd-feature
      :link: wan21
      :link-type: doc

      Bidirectional Wan 2.1 — T2V 1.3B / 480p and I2V 14B / 480p.
      The parity baseline for ``self-forcing`` and
      ``causal-forcing`` recipes.

   .. grid-item-card:: Cosmos-Predict2.5
      :class-card: fd-feature
      :link: cosmos_predict2
      :link-type: doc

      Bidirectional Cosmos-Predict2 recipes (T2V / I2V, 2B).

Adding your own model
---------------------

.. container:: fd-eyebrow

   The integration surface

.. container:: fd-split fd-split-reverse

   .. container:: fd-split-text

      Recipes ship in-tree under ``integrations/`` as workspace-member
      packages, or out-of-tree as third-party plugins discovered
      through entry points. The :doc:`integration guide
      </developer_guides/new_integration>` walks the full surface
      — what to subclass, what to register, where the parity test
      goes.

      The shipping recipes are the best reference: each
      ``integrations/<recipe>/`` directory carries the runner, the
      registration shim, and (where applicable) a
      ``tests/parity_check/`` script that runs the same generation
      under the upstream library.

   .. container:: fd-split-visual

      .. container:: fd-info-card

         .. container:: fd-info-card-title

            Integration checklist

         | 1. Subclass the appropriate runner base class.
         | 2. Register the recipe via the ``flashdreams.runners``
         |    entry point.
         | 3. Add a per-model page under ``docs/source/models/``.
         | 4. (Streaming) wire ``--total-blocks`` into the runner
         |    config.
         | 5. (Parity) ship a ``tests/parity_check/run.sh``.

.. toctree::
   :hidden:
   :maxdepth: 1

   omnidreams
   self_forcing
   causal_forcing
   causal_wan22
   lingbot_world
   flashvsr
   cosmos_predict2
   wan21
