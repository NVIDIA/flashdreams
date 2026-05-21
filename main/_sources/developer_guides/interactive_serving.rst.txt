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

Interactive serving
===================================

FlashDreams supports interactive serving workflows through integration-specific
integrations, with Lingbot-World as the primary reference implementation.

What this guide covers
----------------------

- launching serving-oriented runners,
- wiring serving state/control inputs into runner configuration,
- validating single-GPU and multi-GPU launch patterns.

Lingbot-World serving baseline
------------------------------

Single GPU:

.. code-block:: bash

   uv run flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21

Multi GPU:

.. code-block:: bash

   uv run torchrun --nproc_per_node=2 --no-python flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21

Serving implementation references
---------------------------------

- :doc:`/models/lingbot_world` for model-specific launch options.
- :doc:`/apis/serving` for serving API concepts and component mapping.
- ``integrations/lingbot/lingbot/webrtc`` for the WebRTC serving stack.
