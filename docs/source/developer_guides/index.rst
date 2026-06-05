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

These guides cover how the system is structured underneath the CLI:
the inference pipeline a recipe runs through, the configuration layer
every recipe shares, the integration surface for adding a new method,
common patterns for driving the pipeline from Python, and the shape
of an interactive serving session. They are conceptual; the
:doc:`/api/index` is the per-symbol reference.

The pipeline overview is the anchor for the rest. The config system
is the layer every recipe shares; new integrations sit on top of
both. Usage patterns and interactive serving describe how the
pipeline is embedded in surrounding code.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Inference pipeline overview
      :link: inference_pipeline_overview
      :link-type: doc

      The end-to-end computation flow: warmup, CUDA-graph capture,
      the autoregressive-step body, the ring-attention shard group,
      and finalize. The mental model the rest of the project assumes.

   .. grid-item-card:: Config system
      :link: config_system
      :link-type: doc

      How every overridable field is surfaced as a CLI flag, how
      recipe defaults compose, and how to layer overrides on top.

   .. grid-item-card:: Add a new method
      :link: new_integration
      :link-type: doc

      The entry-point surface a new recipe ships against: what to
      subclass, what to register, and where the parity tests live.

   .. Temporarily commented out for internal development:
   .. grid-item-card:: Usage patterns
   ..    :link: usage_patterns
   ..    :link-type: doc
   ..
   ..    Common ways to drive FlashDreams from Python: the CLI, the
   ..    in-process runner API, and the pipeline-level surface for
   ..    embedding.
   ..
   .. grid-item-card:: Interactive serving
   ..    :link: interactive_serving
   ..    :link-type: doc
   ..
   ..    Keeping a streaming session alive: warmup, steady-state
   ..    generation, and how the WebRTC and gRPC servers under
   ..    ``integrations/`` wire the pipeline up.

Where these guides fit
----------------------

Working forward from a recipe, start with the pipeline overview,
then read the recipe's per-model page under :doc:`/models/index`,
then drop into the matching module under :doc:`/api/index` for the
implementation details. The :doc:`/quickstart/index` covers the
two-command path from install to a generated clip.

.. toctree::
   :hidden:
   :maxdepth: 1

   inference_pipeline_overview
   config_system
   new_integration
.. Temporarily commented out for internal development:
..   usage_patterns
..   interactive_serving
