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

Recipes
===================================

Concrete model implementations live under ``flashdreams.recipes``. Each
recipe wires a checkpoint family into the
:doc:`infra <infra>` abstractions and exposes the resulting
``StreamInferencePipelineConfig`` factories that the example launchers
consume.

.. note::

   Recipe modules import the heavy GPU stack (transformer-engine, CUDA
   ops) at import time, so this page shows them by *automodule* with
   ``:no-undoc-members:`` to keep the rendered API focused on the names
   that recipes actually expose. The unified ``flashdreams-run`` CLI
   shows end-to-end usage; see :doc:`/examples/alpadreams` and friends.

AlpaDreams
----------

.. automodule:: flashdreams.recipes.alpadreams
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: flashdreams.recipes.alpadreams.config
   :members:
   :no-undoc-members:
   :show-inheritance:

.. automodule:: flashdreams.recipes.alpadreams.pipeline
   :members:
   :no-undoc-members:
   :show-inheritance:

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
