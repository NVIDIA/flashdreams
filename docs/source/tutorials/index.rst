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

Tutorials
===================================

Hands-on walkthroughs that take a new user from a freshly cloned
repository to their first generated video, then deeper into the
recipes, the multi-GPU launcher, and CUDA-graph capture.

If you are looking for the API surface or the headline performance
numbers, see the :doc:`API Reference <../api/index>` and the
:doc:`Benchmarks <../benchmarks/index>` pages. If you are looking to
contribute or to ask a question, head to :doc:`Community
<../community/index>`.

.. admonition:: New to streaming inference?
   :class: fd-callout

   Start with :doc:`quickstart` to install the workspace, then run
   :doc:`your-first-generation` to drive a Wan2.1 T2V runner end to
   end. Both fit in well under an hour on a single GPU.

Getting started
---------------

.. grid:: 1 2 2 3
   :gutter: 3
   :margin: 0 0 4 0

   .. grid-item-card:: Install in 60 seconds
      :class-card: fd-feature
      :link: quickstart
      :link-type: doc

      ``uv sync`` the workspace, install the runner extras, and verify
      the CLI with ``flashdreams-run --help``.

   .. grid-item-card:: Your first generation
      :class-card: fd-feature
      :link: your-first-generation
      :link-type: doc

      A guided run of the ``wan21-t2v-1.3b-480p`` runner with prompt
      overrides and an annotated walk-through of the output.

   .. grid-item-card:: Pick a model
      :class-card: fd-feature
      :link: tutorials-per-recipe-walkthroughs
      :link-type: ref

      Jump straight to a per-recipe walkthrough. Streaming AR models
      and bidirectional models are listed below.

.. toctree::
   :hidden:
   :maxdepth: 1

   quickstart
   your-first-generation

.. _tutorials-per-recipe-walkthroughs:

Per-recipe walkthroughs
-----------------------

Each recipe ships with its own end-to-end walkthrough under
``docs/source/examples/``. The CLI slug each page documents is a real
``flashdreams-run`` subcommand backed by a
:class:`~flashdreams.infra.runner.RunnerConfig` literal in the recipe
package.

.. toctree::
   :maxdepth: 1
   :caption: Autoregressive models

   AlpaDreams <../examples/alpadreams>
   Self-Forcing <../examples/self_forcing>
   Causal-Forcing <../examples/causal_forcing>
   FastVideo Causal Wan2.2 <../examples/causal_wan22>
   Lingbot-World camera-control I2V <../examples/lingbot_world>

.. toctree::
   :maxdepth: 1
   :caption: Bidirectional models

   Wan2.1 (T2V / I2V) <../examples/wan21>

A one-line preview:

- **AlpaDreams** — driving-scene video generation; single-view and
  multi-view runners with HDMap conditioning.
- **Self-Forcing** — out-of-tree plugin recipe for the upstream
  Self-Forcing T2V checkpoint.
- **Causal-Forcing** — chunk-wise and frame-wise variants in T2V and
  I2V; runs on top of the Wan2.1 1.3B backbone.
- **FastVideo Causal Wan2.2** — 14B T2V via the FastVideo Wan2.2
  causal recipe.
- **Lingbot-World** — fast camera-control I2V driven by the
  Lingbot-World upstream.
- **Wan2.1 (bidirectional)** — reference single-step Wan2.1 in T2V
  (1.3B) and I2V (14B), used as the parity baseline.

Going deeper
------------

Once a single runner works end to end, these short pages cover the
cross-cutting performance features the streaming pipeline exposes and
the path for plugging in a custom recipe.

.. toctree::
   :maxdepth: 1
   :caption: Advanced topics

   advanced-cuda-graphs
   advanced-distributed
   advanced-custom-recipe
   ../developer_guides/new_recipes

Where next
----------

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 0 0 4 0

   .. grid-item-card:: Benchmarks
      :class-card: fd-feature
      :link: ../benchmarks/index
      :link-type: doc

      Headline throughput and latency numbers, plus the recipes they
      were measured against.

   .. grid-item-card:: API reference
      :class-card: fd-feature
      :link: ../api/index
      :link-type: doc

      The autodoc surface for :mod:`flashdreams.core`,
      :mod:`flashdreams.infra`, :mod:`flashdreams.recipes`, and the
      serving layer.

   .. grid-item-card:: Add a new recipe
      :class-card: fd-feature
      :link: ../developer_guides/new_recipes
      :link-type: doc

      Long-form guide to scaffolding a custom streaming recipe in its
      own repository and registering it with ``flashdreams-run``.

   .. grid-item-card:: Community
      :class-card: fd-feature
      :link: ../community/index
      :link-type: doc

      How to ask questions, file bugs, and contribute back.
