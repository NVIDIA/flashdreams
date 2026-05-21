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

:orphan:

Models
===================================

FlashDreams supports multiple world/video model families behind one unified CLI.

Running a model
---------------

.. code-block:: bash

   uv run flashdreams-run <MODEL_SLUG> --help

Examples:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash --total-blocks 7
   uv run flashdreams-run lingbot-world-fast --example-data True --total-blocks 21

Implemented models
------------------

- :doc:`OmniDreams </models/omnidreams>`
- :doc:`Self-Forcing </models/self_forcing>`
- :doc:`Causal-Forcing </models/causal_forcing>`
- :doc:`Causal Wan2.2 </models/fastvideo_wan22>`
- :doc:`Lingbot-World </models/lingbot_world>`
- :doc:`Wan2.1 </models/wan21>`

Adding your own model
---------------------

See :doc:`/developer_guides/new_recipes` for model integration and registration
guidance.
