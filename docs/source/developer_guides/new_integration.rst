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

Add a new method
===================================

Before you start adding a new method, we highly recommend reading the :doc:`/developer_guides/inference_pipeline_overview` and :doc:`/developer_guides/config_system` pages first to obtain an overview of the system architecture.

FlashDreams aims to offer researchers a codebase that they can utilize to extend and develop novel video and world models. Our vision is for users to establish a *standalone repository* that imports FlashDreams as a dependency and overrides pipeline components (such as encoders, transformers, or decoders) to cater to specific functionality requirements of the new approach. We encourage you to maintain your method externally rather than pushing changes directly into the `integrations/ <https://github.com/NVIDIA/flashdreams/tree/main/integrations>`_ directory of this repository.

However, if any of your new features require modifications to the core FlashDreams infra or introduce generally useful components (such as the :mod:`TAEHV decoder <flashdreams.recipes.taehv>`), we encourage you to submit a PR to enable others to benefit from them.

File structure
--------------

We recommend the following file structure for your new method:

.. code-block:: text

    customized_method/
    ├── customized_method/
    │   ├── __init__.py
    │   ├── runner.py            # Custom runner and runner config
    │   ├── config.py            # Pipeline config literals
    │   ├── pipeline.py          # [optional] custom pipeline behavior
    │   ├── transformer.py       # [optional] DiT network for diffusion.
    │   ├── encoder.py           # [optional] custom control encoder
    │   ├── decoder.py           # [optional] custom outputdecoder
    │   └── ...
    └── pyproject.toml

Add optional files only when you need to customize the behavior beyond what is available in FlashDreams.
Most integrations can use the base :class:`~flashdreams.infra.pipeline.StreamInferencePipeline` directly and only provide model components plus config literals.
As explained in :doc:`/developer_guides/config_system`, a method typically defines a pipeline config and a runner config.
The runner handles CLI-facing I/O and runtime loops.

.. code-block:: python
   :caption: customized_method/runner.py

   from dataclasses import dataclass, field
   from flashdreams.infra.pipeline import StreamInferencePipeline
   from flashdreams.infra.runner import Runner, RunnerConfig

   @dataclass(kw_only=True)
   class CustomizedMethodRunnerConfig(RunnerConfig):
       """Runner config for my method."""
       _target: type["CustomizedMethodRunner"] = field(default_factory=lambda: CustomizedMethodRunner)
       prompt: str = "A cat surfing."
       total_blocks: int = 60

   class CustomizedMethodRunner(Runner[CustomizedMethodRunnerConfig, StreamInferencePipeline]):
       def run(self) -> None:
           cfg = self.config

           # 1. Initialize the autoregressive cache
           cache = self.pipeline.initialize_cache(text=[cfg.prompt])

           # 2. Drive the autoregressive rollout
           for i in range(cfg.total_blocks):
               video_chunk = self.pipeline.generate(autoregressive_index=i, cache=cache)
               self.pipeline.finalize(autoregressive_index=i, cache=cache)

           if self.is_rank_zero:
               # 3. Write outputs only on the main process
               ...

.. code-block:: python
   :caption: customized_method/config.py

   from flashdreams.infra.diffusion.model import DiffusionModelConfig
   from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
   from flashdreams.infra.pipeline import StreamInferencePipelineConfig
   from flashdreams.infra.runner import RunnerConfig

   from customized_method.runner import CustomizedMethodRunnerConfig
   from customized_method.transformer import MyTransformerConfig

   # Define your pipeline config literal
   CUSTOMIZED_PIPELINE_CONFIG = StreamInferencePipelineConfig(
       name="customized-method",
       diffusion_model=DiffusionModelConfig(
           transformer=MyTransformerConfig(),
           scheduler=FlowMatchSchedulerConfig(),
       ),
   )

   # Define your runner config
   CUSTOMIZED_METHOD_RUNNER = CustomizedMethodRunnerConfig(
       runner_name=CUSTOMIZED_PIPELINE_CONFIG.name,
       description="Custom description for my method.",
       pipeline=CUSTOMIZED_PIPELINE_CONFIG,
   )


You can use the existing integrations under the `integrations/ <https://github.com/NVIDIA/flashdreams/tree/main/integrations>`_ directory as a minimal guide. These folders are simple examples of what mini standalone repositories that depend on FlashDreams look like. Examples are often the best way to learn; take a look at the `LingBot-World <https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot>`_ and `Self-Forcing <https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing>`_ integrations for good references on how to extend and use FlashDreams in your own projects.

Registering your method
-----------------------

After registering your method you should be able to see it in the CLI helptext and run it, leveraging the existing CLI for complete command line control:

.. code-block:: bash

   # List all available models, including your new one
   flashdreams-run --help

   # See configurable parameters for your new model
   flashdreams-run customized-method --help

   # Run the new model
   flashdreams-run customized-method --prompt "A beautiful custom generation."

To register your own models, package your code as a Python package and declare a runner entry point in its ``pyproject.toml``. FlashDreams discovers every registered runner at import time and surfaces it through the same command-line interface as the in-tree integrations. There is no central manifest to keep in sync.

Create a ``pyproject.toml`` file. This is where the entrypoint to your method is set and also where you can specify additional dependencies required by your codebase.

.. code-block:: toml
   :caption: pyproject.toml

   [project]
   name = "customized_method"
   version = "0.1.0"

   dependencies = [
       "flashdreams", # consider pinning the version, ie "flashdreams==0.1.0"
       "mediapy>=1.1",
   ]

   [tool.setuptools.packages.find]
   include = ["customized_method*"]

   [project.entry-points."flashdreams.runner_configs"]
   customized-method = "customized_method.config:CUSTOMIZED_METHOD_RUNNER"

Finally, run the following to register the method:

.. code-block:: bash

   pip install -e .

When developing a new method, you don't always want to install your code as a package. Instead, you may use the ``FLASHDREAMS_RUNNER_CONFIGS`` environment variable to temporarily register your custom method.

.. code-block:: bash

   export FLASHDREAMS_RUNNER_CONFIGS="customized-method=customized_method.config:RUNNER_CONFIGS"

The ``FLASHDREAMS_RUNNER_CONFIGS`` environment variable additionally accepts a function (a zero-arg callable) to temporarily register your custom runner if construction has side effects you want to defer until CLI time.

Adding to the FlashDreams documentation
---------------------------------------

We invite researchers to contribute their own integrations to our official codebase and documentation. You can find more information on how to do this in the repository's `CONTRIBUTING.md <https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md>`_.
See the existing model pages under the `docs/source/models/ <https://github.com/NVIDIA/flashdreams/tree/main/docs/source/models>`_ directory (e.g., :doc:`/models/self_forcing`) as templates for documenting your new method.
