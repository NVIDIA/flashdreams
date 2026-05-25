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

Launch your first model
===================================

This page provides a minimal path for:

1. offline / batch-like long-rollout model inference with Self-Forcing,
2. online interactive world-model serving with LingBot-World.

If you are new to the distinction, read
:doc:`/developer_guides/offline_vs_online` first.

Prerequisites
-------------

Complete the setup in :doc:`/quickstart/installation` first:

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
     Run the commands above, then use the model catalog for variants,
     upstream links, and performance notes.
   </div>

Run model serving (LingBot-World)
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
     Use this quick path to validate serving. The developer guide explains
     the serving session model and implementation references.
   </div>

Next steps
----------

- For complete per-model launch options, see :doc:`/models/index`.
- For model-specific details, see :doc:`/models/index`.
- For the conceptual difference between one-shot inference and persistent
  serving, see :doc:`/developer_guides/offline_vs_online`.
