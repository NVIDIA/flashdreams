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

API reference
=============

FlashDreams is organised as four cooperating layers. **Core** holds the
low-level kernels and process-group helpers; **Infra** defines the
swappable abstractions (encoder, diffusion model, decoder, pipeline)
that every model plugs into; **Recipes** wires concrete checkpoints into
those abstractions; and **Serving** fronts the deployable surfaces. The
:doc:`landing page </index>` sketches the same picture at a higher level
— start there if you want the motivation before the signatures.

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 0 0 4 0

   .. grid-item-card:: Core
      :class-card: fd-feature
      :link: ../apis/core
      :link-type: doc

      Low-level kernels and process-group utilities — attention,
      block-structured KV cache, distributed init.

      Read the reference →

   .. grid-item-card:: Infra
      :class-card: fd-feature
      :link: ../apis/infra
      :link-type: doc

      The swappable abstractions every recipe plugs into — config,
      encoder, diffusion model, decoder, and the streaming pipeline.

      Read the reference →

   .. grid-item-card:: Recipes
      :class-card: fd-feature
      :link: ../apis/recipes
      :link-type: doc

      Concrete model implementations that wire checkpoint families into
      the infra abstractions and expose pipeline factories.

      Read the reference →

   .. grid-item-card:: Serving
      :class-card: fd-feature
      :link: ../apis/serving
      :link-type: doc

      Deployable surfaces for shipping a recipe behind a service
      boundary.

      Read the reference →

Where to start
--------------

- **"I want to add a new model."** Read the
  :doc:`new-recipes developer guide </developer_guides/new_recipes>`
  for the step-by-step on plugging a checkpoint family into the infra
  abstractions, then browse :doc:`/apis/recipes` for the shape of the
  in-tree recipes.
- **"I want to understand streaming inference."** Open
  :doc:`/apis/infra` and read the *Pipeline* section — the
  ``StreamInferencePipeline.generate`` method docstring and body trace
  the per-AR-step loop through encoder, diffusion model, and decoder.
- **"I want to drop into a kernel."** Head straight to
  :doc:`/apis/core` — the attention classes and ``BlockKVCache`` are
  the surface you'll touch most often.
- **"I want a worked example."** Start with
  :doc:`/tutorials/index` for the install-to-first-generation
  walkthrough plus the advanced topics (CUDA graphs, distributed
  launching, custom recipes).

.. toctree::
   :maxdepth: 1
   :caption: API reference
   :hidden:

   ../apis/core
   ../apis/infra
   ../apis/recipes
   ../apis/serving
