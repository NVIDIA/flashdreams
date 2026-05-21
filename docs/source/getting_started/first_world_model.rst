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

Launch your first world model
===================================

This page provides a minimal path for:

1. long-rollout model inference (Self-Forcing),
2. interactive world-model serving (Lingbot-World).

Prerequisites
-------------

Complete the setup in :doc:`/getting_started/installation` first:

- :ref:`run-models-directly-in-this-codebase`
- :ref:`environment-variables`

Run model inference (Self-Forcing)
----------------------------------

Single GPU:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash --total-blocks 7

Multi GPU (context parallel):

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b-flash --total-blocks 7

.. raw:: html

   <div class="video-slot">
     <strong>Inference walkthrough</strong><br>
     Run the commands above, then compare output and runtime knobs in the
     corresponding page under :doc:`/models/index`.
   </div>

Run model serving (Lingbot-World)
---------------------------------

Single GPU:

.. code-block:: bash

   uv run flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21

Multi GPU:

.. code-block:: bash

   uv run torchrun --nproc_per_node=2 --no-python flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21

.. raw:: html

   <div class="video-slot">
     <strong>Serving walkthrough</strong><br>
     For full serving architecture and deployment guidance, see
     :doc:`/developer_guides/interactive_serving`.
   </div>

Next steps
----------

- For complete per-model launch options, see :doc:`/getting_started/supported_models`.
- For model-specific details, see :doc:`/models/index`.
