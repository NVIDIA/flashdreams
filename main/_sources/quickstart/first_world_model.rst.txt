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

1. offline / batch-like long-rollout model inference with Self-Forcing.
2. online interactive world-model serving with LingBot-World.

Prerequisites
-------------

Complete the setup in :doc:`/quickstart/installation` first.

Run Self-Forcing T2V offline inference
--------------------------------------

Launch an offline inference run using the :doc:`Self-Forcing </models/self_forcing>` model:

.. code-block:: bash

   uv run --project integrations/self_forcing \
       flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash \
       --total-blocks 7

Run LingBot-World interactive server
------------------------------------

Launch an interactive serving session using the :doc:`LingBot-World </models/lingbot_world>` model:

.. code-block:: bash

   uv run --project integrations/lingbot \
       flashdreams-run lingbot-world-fast \
       --example-data True \
       --total-blocks 21

Next steps
----------

Explore models:

- :doc:`/models/index` - Browse all supported models, their specific launch commands, and configurations.

For developers:

- :doc:`/developer_guides/inference_pipeline_overview` - Learn about the system architecture and generation loop.
- :doc:`/developer_guides/config_system` - Understand how to modify pipeline and runner configurations.
- :doc:`/developer_guides/new_integration` - Guide to adding your own custom models and methods.
- :doc:`/api/index` - Check the Python API and CLI references.
