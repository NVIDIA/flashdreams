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

FlashDreams
===================================

.. raw:: html

   <style>
     #furo-main-content > section > h1 { display: none; }
   </style>
   <div class="homepage-logo-wrap">
     <img src="_static/flashdreams_logo_horizontal.png" alt="FlashDreams">
   </div>

FlashDreams is a high-performance streaming inference stack for world and
video models. It focuses on long-rollout autoregressive generation, efficient
multi-GPU execution, and practical model serving.

Highlights
----------

- FlashDreams is built for **streaming long-rollout world-model inference**
  with per-step cache updates and low overhead in autoregressive loops.
- The unified ``flashdreams-run`` CLI exposes both built-in and plugin models
  behind a single launch interface, from Self-Forcing and Causal-Forcing to
  Lingbot-World and Causal Wan2.2.
- FlashDreams is designed for **multi-GPU context-parallel execution** with
  torchrun-based scaling and integration-level support for efficient transformer
  attention/cache pipelines.
- The framework is **modular and extensible**: pipelines and runner configs can
  be added as external integration packages via entry points without forking
  core infra.
- Current benchmarks show strong practical speedups in matched environments,
  including up to **2.49x** on Lingbot-World (H100, 4xGPU) and up to
  **1.95x** on Self-Forcing (GB200, block-6 total latency) against official
  baselines.

.. raw:: html

   <div class="video-slot">
     <strong>Project overview media</strong><br>
     See model-specific pages under ``Models`` for runnable commands and
     available qualitative assets.
   </div>

Quick install
-------------

.. code-block:: bash

   # Library usage
   pip install flashdreams

   # Latest main branch
   pip install "git+https://github.com/NVIDIA/flashdreams.git"

   # Codebase workflow
   git clone https://github.com/NVIDIA/flashdreams.git
   cd flashdreams
   uv sync --extra dev --extra runners
   uv run flashdreams-run --help

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: getting_started/index
      :link-type: doc

      Installation, first model launch, and supported model overview.

   .. grid-item-card:: Developer Guides
      :link: developer_guides/index
      :link-type: doc

      Architecture, model integration, configs, and serving guidance.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      CLI usage and API surfaces.

   .. grid-item-card:: Models
      :link: models/index
      :link-type: doc

      Model catalog with per-model run commands and links.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   getting_started/installation
   getting_started/first_world_model
   getting_started/supported_models

.. toctree::
   :maxdepth: 1
   :caption: Developer Guides
   :hidden:

   developer_guides/new_recipes
   developer_guides/system_overview
   developer_guides/configs
   developer_guides/interactive_serving

.. toctree::
   :maxdepth: 1
   :caption: Models
   :hidden:

   models/omnidreams
   models/self_forcing
   models/causal_forcing
   models/fastvideo_wan22
   models/lingbot_world
   models/wan21

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   CLI <reference/cli>
   API <apis/index>
