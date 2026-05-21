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

System overview
===================================

FlashDreams is organized around a small inference chassis that integration
packages plug into.

.. figure:: /_static/diagrams/system_overview_flow.svg
   :alt: High-level FlashDreams system flow from CLI to pipeline and outputs.

   High-level flow across CLI, runner, pipeline, diffusion/model components,
   and distributed runtime.

Core concepts
-------------

- **Integration**: standalone package under ``integrations/<name>/`` that ships
  pipeline configs + runner configs together.
- **Pipeline**: runtime composition of integration components for generation and
  cache updates.
- **Runner**: user-facing launcher that exposes runtime I/O and overrides as
  CLI flags.
- **Registry**: merged map of built-in and plugin-discovered runner configs.

Execution flow (code-mapped)
----------------------------

1. **CLI + registry**: ``flashdreams-run`` dispatches through the runner
   registry from ``flashdreams/scripts/cli.py`` and
   ``flashdreams/configs/runner_configs.py``.
2. **Runner setup**: the selected runner config instantiates
   ``Runner`` (``flashdreams/infra/runner.py``), builds the pipeline, and
   prepares runtime inputs.
3. **Pipeline loop**: ``StreamInferencePipeline`` in
   ``flashdreams/infra/pipeline/base.py`` executes
   ``initialize_cache`` -> ``generate`` -> ``finalize`` across AR steps.
4. **Model internals**: the diffusion model runs scheduler + transformer passes;
   integration modules provide concrete network/encoder/decoder implementations.
5. **Distributed execution**: for multi-GPU runs, ``torchrun --no-python``
   defines world size/rank, and integration transformers derive context-parallel
   behavior from ``torch.distributed``.
6. **Persistence**: user-facing artifacts (video/stats/logs) are written by
   rank 0 to avoid duplicated outputs.

Code map
--------

- :doc:`/apis/core` for low-level kernels and distributed utilities.
- :doc:`/apis/infra` for pipeline, diffusion, and runner abstractions.
- :doc:`/apis/recipes` for pipeline/runner API surfaces and integration map.
- :doc:`/developer_guides/new_recipes` for implementing and registering new
  models.
