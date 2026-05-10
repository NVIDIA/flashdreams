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

flashdreams
===================================

Overview
--------

*flashdreams* is a streaming inference pipeline for diffusion-based video
generation. It targets autoregressive ("self-forcing" / "causal-forcing")
flow-matching models with first-class support for KV-cached transformers,
ring attention across context-parallel ranks, and CUDA-graph capture for
the steady-state forward, plus a single bidirectional reference model
for parity testing.

The library is organised around a few sharp abstractions
(:doc:`apis/infra`) that every recipe (:doc:`apis/recipes`) plugs into;
shared low-level kernels and distributed helpers live under
:doc:`apis/core`. The unified ``flashdreams-run`` CLI fronts every
shipped recipe; per-recipe usage is walked through in the sections below.

Installation
------------

The repository is a `uv <https://docs.astral.sh/uv/>`_ workspace:

.. code-block:: bash

   uv sync --extra dev
   uv run pytest flashdreams/tests

The ``flashdreams-run`` CLI runners lazy-import ``mediapy`` + ``opencv``
for image / video I/O; install the ``runners`` extra to enable them:

.. code-block:: bash

   uv sync --extra dev --extra runners
   uv run flashdreams-run --help

See the project ``README.md`` for the full container-based workflow on a
Slurm node.

.. toctree::
   :maxdepth: 1
   :caption: Supported Autoregressive Models

   examples/alpadreams
   examples/self_forcing
   examples/causal_forcing
   examples/causal_wan22
   examples/lingbot_world

.. toctree::
   :maxdepth: 1
   :caption: Supported Bidirectional Models

   examples/wan21

.. toctree::
   :maxdepth: 1
   :caption: FlashDreams Inference API

   apis/core
   apis/infra
   apis/recipes

.. toctree::
   :maxdepth: 1
   :caption: FlashDreams Serving API

   apis/serving

.. toctree::
   :maxdepth: 1
   :caption: Developer Guides

   developer_guides/new_recipes
