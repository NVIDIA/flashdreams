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

CLI and API Reference
=====================

Reference pages for the FlashDreams command-line interface and Python APIs.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: CLI
      :link: cli
      :link-type: doc

      The unified ``flashdreams-run`` entry point: listing runner slugs,
      inspecting a runner's options, and launching single- and multi-GPU
      inference.

   .. grid-item-card:: Core
      :link: core
      :link-type: doc

      The low-level kernels and process-group utilities that
      integrations share: attention, the block-structured KV cache, and
      distributed helpers.

   .. grid-item-card:: Infra
      :link: infra
      :link-type: doc

      The swappable abstractions every integration plugs into: the
      config system, the encoder / diffusion-model / decoder triple, and
      the streaming inference pipeline that drives them.

   .. grid-item-card:: Pipelines and runners
      :link: integrations
      :link-type: doc

      The two public layers a model integration is built from: pipelines
      that define model behavior and runners that define CLI-facing I/O.

   .. grid-item-card:: Serving
      :link: serving
      :link-type: doc

      The runner / pipeline building blocks for integration-driven
      serving, with LingBot-World as the canonical interactive-transport
      reference.

.. toctree::
   :hidden:
   :maxdepth: 1

   cli
   core
   infra
   integrations
   serving
