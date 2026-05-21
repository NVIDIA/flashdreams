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

Installation
===================================

Choose the setup path that matches your goal.

Use FlashDreams as a library
----------------------------

Install from PyPI:

.. code-block:: bash

   pip install flashdreams

Install the latest main branch:

.. code-block:: bash

   pip install "git+https://github.com/NVIDIA/flashdreams.git"

.. _run-models-directly-in-this-codebase:

Run models directly in this codebase
------------------------------------

Clone the repository and use the workspace ``uv`` environment:

.. code-block:: bash

   git clone https://github.com/NVIDIA/flashdreams.git
   cd flashdreams
   uv sync --extra dev --extra runners

The unified runner CLI is available through ``uv run``:

.. code-block:: bash

   uv run flashdreams-run --help

.. _environment-variables:

Environment variables
---------------------

Most model runs need Hugging Face authentication:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>
   export HF_HOME=~/.cache/huggingface  # optional

For more environment and container details, see the project
`README <https://github.com/NVIDIA/flashdreams/blob/main/README.md>`_ and
the model pages under :doc:`/models/index`.
