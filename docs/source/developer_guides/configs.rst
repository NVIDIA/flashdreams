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

Configs
===================================

FlashDreams configuration is intentionally layered so users can start from a
shipped runner slug and only override what they need.

Configuration layers
--------------------

1. **Pipeline config** defines model and generation defaults for an integration.
2. **Runner config** wraps a pipeline config and adds user-facing I/O fields.
3. **CLI overrides** from ``flashdreams-run <slug> --help`` are applied at run
   time through tyro.

Typical usage
-------------

Inspect all available fields for a runner:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash --help

Override selected nested fields:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash \
       --total-blocks 7 \
       --pipeline.diffusion-model.transformer.use-cuda-graph True

Resolve a fully merged config without running inference:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash --no-instantiate

Authoring config variants
-------------------------

- Use :func:`flashdreams.infra.config.derive_config` to create concise variants
  from existing pipeline configs.
- Keep one ``RunnerConfig`` literal per shipped slug in ``config.py``.
- Ensure each runner has a clear ``description``; this is user-facing in
  ``flashdreams-run --help``.

For end-to-end examples, see :doc:`/developer_guides/new_recipes`.
