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

CLI
===================================

FlashDreams exposes a unified command line entry point:
``flashdreams-run``.

Core commands
-------------

List all available runner slugs:

.. code-block:: bash

   uv run flashdreams-run --help

Inspect one runner's full options:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-taehv --help

Run a single-GPU inference:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-taehv --total-blocks 7

Run a multi-GPU inference:

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b-taehv --total-blocks 7

Resolve config only (no model instantiation):

.. code-block:: bash

   uv run flashdreams-run --no-instantiate self-forcing-wan2.1-t2v-1.3b-taehv

See also
--------

- :doc:`/quickstart/first_world_model`
- :doc:`/developer_guides/config_system`
- :doc:`/api/infra`
