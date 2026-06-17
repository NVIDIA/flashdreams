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

How to use FlashDreams as a developer
=====================================

Pick the path that matches your goal. For the full runtime map, see
:doc:`/developer_guides/inference_pipeline_overview`.

.. raw:: html

   <div class="fd-highlight-grid">
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Run existing models</div>
       <div class="fd-highlight-body">Clone the repo, install runner extras, then jump to a model page for the exact slug and flags.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Use as a Python library</div>
       <div class="fd-highlight-body">Install FlashDreams and import runtime contracts from <code>flashdreams.infra</code>.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Add a standalone model</div>
       <div class="fd-highlight-body">Ship configs and runners from your own package through <code>flashdreams.runner_configs</code>.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Build serving apps</div>
       <div class="fd-highlight-body">Keep sessions alive and stream outputs through the serving path.</div>
     </div>
   </div>

Run existing models from source
-------------------------------

.. code-block:: bash

   git clone https://github.com/NVIDIA/flashdreams.git
   cd flashdreams
   uv sync --extra dev --extra runners
   uv run flashdreams-run --help

Then pick a model page for actual slugs and flags:

- :doc:`/models/omnidreams`
- :doc:`/models/self_forcing`
- :doc:`/models/causal_forcing`
- :doc:`/models/causal_wan22`
- :doc:`/models/lingbot_world`
- :doc:`/models/wan21`
- :doc:`/models/flashvsr`
- :doc:`/models/cosmos_predict2`

Programmatic access
-------------------

.. code-block:: bash

   pip install flashdreams

.. code-block:: python

   from flashdreams.infra.pipeline import StreamInferencePipeline
   from my_integration.config import MY_MODEL_RUNNER

   runner = MY_MODEL_RUNNER.setup()
   runner.run()

For lower-level experiments:

.. code-block:: python

   from my_integration.config import MY_PIPELINE

   pipeline: StreamInferencePipeline = MY_PIPELINE.setup()
   cache = pipeline.initialize_cache(height=480, width=832)

   for ar_idx in range(4):
       output = pipeline.generate(ar_idx, cache)
       pipeline.finalize(ar_idx, cache)

Arguments are model-specific; use the integration runner as the source of truth.

Add a new model from a standalone repo
--------------------------------------

Minimal entry point:

.. code-block:: toml

   [project]
   name = "my-flashdreams-model"
   dependencies = ["flashdreams"]

   [project.entry-points."flashdreams.runner_configs"]
   my-model-fast = "my_integration.config:MY_MODEL_FAST_RUNNER"

Then:

.. code-block:: bash

   pip install -e .
   flashdreams-run my-model-fast --help

Use :doc:`/developer_guides/new_integration` for the complete authoring guide.

Next links
----------

- :doc:`/models/index` for model commands.
- :doc:`/developer_guides/new_integration` for integration authoring.

.. - :doc:`/developer_guides/interactive_serving` for serving concepts.
