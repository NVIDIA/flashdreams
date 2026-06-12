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

Pipelines and runners
===================================

FlashDreams model integrations are built from two public layers:

- **Pipelines** (``StreamInferencePipelineConfig``) that define model behavior.
- **Runners** (``RunnerConfig`` + ``Runner``) that define CLI-facing I/O.

Most actively developed model implementations now live under ``integrations/*``
as plugin-style standalone packages. This page keeps documenting the in-tree
pipeline modules that are still exposed from ``flashdreams.recipes``.

.. note::

   Pipeline modules import the heavy GPU stack (transformer-engine, CUDA
   ops) at import time, so this page shows them by *automodule* with
   ``:no-undoc-members:`` to keep the rendered API focused on the names
   that these in-tree modules actually expose. The unified ``flashdreams-run``
   CLI shows end-to-end usage; see :doc:`/models/index` for model launch
   examples.

Integration structure (current)
-------------------------------

For new model work, follow ``integrations/<name>/``:

- ``config.py``: pipeline + runner config literals (slugged entries).
- ``runner.py``: runtime I/O, cache init, generate/finalize loop, persistence.
- ``pipeline.py`` and ``transformer/*``: model compute path.
- ``pyproject.toml``: plugin packaging + entry-point registration.

This makes each integration effectively a standalone repository while still
plugging into the same ``flashdreams-run`` registry.

Reference integration folders
-----------------------------

- `omnidreams <https://github.com/NVIDIA/flashdreams/tree/main/integrations/omnidreams>`_
- `self_forcing <https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing>`_
- `causal_forcing <https://github.com/NVIDIA/flashdreams/tree/main/integrations/causal_forcing>`_
- `lingbot <https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot>`_
- `wan21 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/wan21>`_
- `fastvideo_causal_wan22 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/fastvideo_causal_wan22>`_
- `flashvsr <https://github.com/NVIDIA/flashdreams/tree/main/integrations/flashvsr>`_
- `cosmos_predict2 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/cosmos_predict2>`_

NVIDIA OmniDreams
-----------------

OmniDreams now ships as a plugin under ``integrations/omnidreams``; it
registers its runners via the ``flashdreams.runner_configs`` entry-point
group and is no longer part of the in-tree ``flashdreams.recipes`` API
surface. See ``integrations/omnidreams/README.md`` for the plugin entry
point and ``flashdreams-run omnidreams-*`` for the user-facing CLI.

Wan
---

.. automodule:: flashdreams.recipes.wan
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: flashdreams.recipes.wan.pipeline
   :members:
   :no-undoc-members:
   :show-inheritance:

TAEHV
-----

.. automodule:: flashdreams.recipes.taehv
   :members:
   :no-undoc-members:
   :show-inheritance:
