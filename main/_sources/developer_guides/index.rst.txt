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
      method defaults compose, and how to layer overrides on top.

   .. grid-item-card:: Runner slugs and demo dispatch
      :link: runner_slugs
      :link-type: doc

      How public runner names are registered, parsed, matched to manifests,
      and dispatched to integration-owned demo launch modes.

   .. grid-item-card:: Add a new method
      :link: new_integration
      :link-type: doc

      The entry-point surface a new method ships against: what to
      subclass, what to register, and where the parity tests live.

   .. grid-item-card:: Local demo benchmarks
      :link: local_benchmarks
      :link-type: doc

      How to run command-backed local benchmarks that capture logs, MP4s,
      metrics, environment metadata, and an HTML report.

Where these guides fit
----------------------

These guides are conceptual. For a specific method, see its per-model
page under :doc:`/models/index`; for the per-symbol reference, see
:doc:`/api/index`; and for the two-command path from install to a
generated clip, see :doc:`/quickstart/index`.

.. toctree::
   :hidden:
   :maxdepth: 1

   inference_pipeline_overview
   config_system
   runner_slugs
   new_integration
   local_benchmarks

..
   Temporarily commented out for internal development:

   usage_patterns
   interactive_serving
