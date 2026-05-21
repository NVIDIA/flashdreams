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

Adding a new model integration
===================================

FlashDreams is designed for researchers to plug new streaming-inference
integrations into the existing chassis without forking the core.

In the current structure, an integration has two key layers:

- A **pipeline** (``StreamInferencePipelineConfig`` + components) that defines
  generation behavior.
- A **runner** (``RunnerConfig`` + ``Runner``) that exposes user-facing CLI I/O
  such as prompts, images, paths, and runtime toggles.

The codebase still contains some legacy names like ``recipe_name`` in config
fields, but the authoring model you should follow is **pipeline + runner**.

Our vision is for users to keep their custom integration in its **own
repository** that depends on ``flashdreams``, then register the
runner with the unified CLI via a Python entry point. If a custom
piece is broadly useful, we welcome a PR upstreaming it.

The in-repo ``integrations/*`` packages are the reference design for this new
structure. Treat each integration folder as an effectively standalone plugin
repo that can be developed and released independently.

File structure
--------------

We recommend the following layout for an external integration package::

    my_recipe/
    ├── my_recipe/
    │   ├── __init__.py
    │   ├── runner.py            # Runner subclass + RunnerConfig dataclass + I/O helpers
    │   ├── config.py            # Pipeline + RunnerConfig literals (entry-point targets)
    │   ├── pipeline.py          # optional: pipeline subclass / cache
    │   ├── transformer/         # network + Transformer subclass + AR cache
    │   ├── encoder/             # optional: control / text / image encoders
    │   ├── decoder.py           # optional: streaming decoder
    │   └── ...
    └── pyproject.toml

Pipeline and runner authoring
-----------------------------

1. **Pipeline config.** Compose a
   :class:`~flashdreams.infra.pipeline.StreamInferencePipelineConfig`
   literal from your transformer / encoder / decoder configs. Use
   :func:`~flashdreams.infra.config.derive_config` to spawn variants
   without copy-pasting fields. ``recipe_name`` is the legacy config key used
   by CLI slug wiring.

2. **Runner subclass + RunnerConfig dataclass.** In ``runner.py``,
   subclass :class:`~flashdreams.infra.runner.RunnerConfig` with the
   I/O fields the CLI should expose (prompt, image path, …) and
   subclass :class:`~flashdreams.infra.runner.Runner` to implement
   :meth:`~flashdreams.infra.runner.Runner.run`: resolve runtime
   inputs, call ``self.pipeline.initialize_cache(...)``, loop
   ``generate`` + ``finalize``, then persist the output on rank 0.
   Mirror existing integration runners under ``integrations/*/*/runner.py``
   for the canonical control flow.

3. **Per-slug runner literals.** In ``config.py``, instantiate one
   :class:`RunnerConfig` literal per shipped variant alongside the
   matching pipeline configs. ``runner_name`` is the
   ``flashdreams-run`` subcommand slug; by convention it mirrors the
   wrapped pipeline's ``recipe_name``. Always set ``description`` —
   it shows up in ``flashdreams-run --help``. These literals are the
   targets the entry-point declarations (next section) point at.

4. **Module-level dict.** Still in ``config.py``, expose a single
   ``MY_INTEGRATION_RUNNERS: dict[str, RunnerConfig]`` keyed by
   ``runner_name`` for programmatic use.

A minimal sketch:

.. code-block:: python

   # my_integration/runner.py
   from dataclasses import dataclass, field

   from flashdreams.infra.runner import Runner, RunnerConfig


   @dataclass(kw_only=True)
   class MyIntegrationRunnerConfig(RunnerConfig):
       """Runner config for the ``my-model`` family."""

       _target: type = field(default_factory=lambda: MyIntegrationRunner)

       prompt: str = "A cat surfing."
       """User-overridable text prompt."""

       num_ar_steps: int = 1


   class MyIntegrationRunner(Runner[MyIntegrationRunnerConfig, "MyPipeline"]):
       def run(self) -> None:
           cfg = self.config
           cache = self.pipeline.initialize_cache(prompt=cfg.prompt)
           for ar_idx in range(cfg.num_ar_steps):
               out = self.pipeline.generate(ar_idx, cache)
               if ar_idx < cfg.num_ar_steps - 1:
                   self.pipeline.finalize(ar_idx, cache)
           if self.is_rank_zero:
               # save out → cfg.output_dir / f"{cfg.runner_name}.<ext>"
               ...

.. code-block:: python

   # my_integration/config.py
   from flashdreams.infra.runner import RunnerConfig
   from my_integration.runner import MyIntegrationRunnerConfig

   MY_PIPELINE_OFFLINE = ...   # the pipeline-config literal

   MY_MODEL_OFFLINE_RUNNER = MyIntegrationRunnerConfig(
       runner_name="my-model-offline",
       description="My integration: offline reference rollout.",
       pipeline=MY_PIPELINE_OFFLINE,
   )

   MY_INTEGRATION_RUNNERS: dict[str, RunnerConfig] = {
       cfg.runner_name: cfg for cfg in (MY_MODEL_OFFLINE_RUNNER,)
   }

Worked end-to-end examples live in this repo under ``integrations/``:
``self_forcing/``, ``causal_forcing/``, ``lingbot/``, ``wan21/``,
``fastvideo_causal_wan22/``, ``omnidreams/``, and ``cosmos_predict2/``.
Each folder contains its own ``pyproject.toml``, pipeline config, and runner
definitions, and should be treated as a standalone plugin-style repository.

Browse the corresponding GitHub folders directly:

- `integrations/self_forcing <https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing>`_
- `integrations/causal_forcing <https://github.com/NVIDIA/flashdreams/tree/main/integrations/causal_forcing>`_
- `integrations/lingbot <https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot>`_
- `integrations/wan21 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/wan21>`_
- `integrations/fastvideo_causal_wan22 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/fastvideo_causal_wan22>`_
- `integrations/omnidreams <https://github.com/NVIDIA/flashdreams/tree/main/integrations/omnidreams>`_
- `integrations/cosmos_predict2 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/cosmos_predict2>`_

Registering the runner with ``flashdreams-run``
-----------------------------------------------

flashdreams discovers external runners through a Python *entry point*
under the ``flashdreams.runner_configs`` group (matches nerfstudio's
``nerfstudio.method_configs`` naming). The discovery layer lives in
:mod:`flashdreams.plugins.registry`.

Add the entry point to your package's ``pyproject.toml``:

.. code-block:: toml

   [project]
   name = "my-model"
   dependencies = [
       "flashdreams",  # consider pinning a version, e.g. "flashdreams==X.Y.Z"
   ]

   [tool.setuptools.packages.find]
   include = ["my_integration*"]

   [project.entry-points."flashdreams.runner_configs"]
   my-model-offline = "my_integration.config:MY_MODEL_OFFLINE_RUNNER"

You can register either a :class:`RunnerConfig` instance directly, or
a zero-arg callable that returns one (handy when construction has side
effects you want to defer until CLI time).

Install the package and the new runner appears in the CLI:

.. code-block:: bash

   pip install -e .
   flashdreams-run --help                        # lists my-model-offline
   flashdreams-run my-model-offline --help       # shows overridable fields
   flashdreams-run my-model-offline --prompt "..."

Built-in runners always win over a same-slug plugin: an external
package cannot silently shadow a shipped integration slug.
:func:`flashdreams.configs.runner_configs.all_runners` layers
plugin-discovered runners on top of the in-tree registry returned by
:func:`flashdreams.configs.registry.supported_runners` via
:func:`~flashdreams.configs.registry.register_runner` with
``source="plugin"``, which logs and skips any slug already present.

Environment-variable backdoor
-----------------------------

When iterating on an integration you don't always want to ``pip install`` it.
Set ``FLASHDREAMS_RUNNER_CONFIGS`` to a comma-separated list of
``slug=module.path:attribute`` pairs and the CLI picks them up at
startup:

.. code-block:: bash

   export FLASHDREAMS_RUNNER_CONFIGS="my-model-offline=my_integration.config:MY_MODEL_OFFLINE_RUNNER"
   flashdreams-run my-model-offline --prompt "..."

The attribute is loaded with
``getattr(import_module(module), attr)``; if it is callable (and not
already a :class:`RunnerConfig`) it is invoked with no arguments to
obtain the config. The ``slug=`` prefix is purely for log readability —
the registry key always comes from ``cfg.runner_name``. Multiple pairs
are separated with commas.

Bad plugin entries are logged and skipped, so a broken third-party
package never takes the CLI down.

Running the new runner
----------------------

Single GPU:

.. code-block:: bash

   flashdreams-run my-model-offline --prompt "A cat surfing."

Multi-GPU via context-parallelism — integration transformers auto-detect the
CP world size from the launcher. ``--no-python`` tells ``torchrun`` to
``execvp`` the console script directly instead of wrapping it in
``python <script>``:

.. code-block:: bash

   torchrun --nproc_per_node=N --no-python flashdreams-run my-model-offline ...

Resolve and inspect the config without running the pipeline:

.. code-block:: bash

   flashdreams-run my-model-offline --no-instantiate

Programmatic access
-------------------

A pipeline that hasn't been wrapped into a runner is still reachable via
its package imports — useful for serving, tests, and notebooks:

.. code-block:: python

   from my_integration.config import MY_PIPELINE_CONFIGS

   pipeline_cfg = MY_PIPELINE_CONFIGS["my-model-offline"]
   pipeline = pipeline_cfg.setup().to("cuda")

Runners are opt-in: only register one when you want a CLI surface.

Adding an integration to the in-tree distribution
-------------------------------------------------

In-tree integrations under ``integrations/<name>/`` are loaded by the workspace
and registered through the same plugin/entry-point machinery. Keep each one
self-contained (own package metadata, configs, runner definitions, tests), and
ensure every shipped runner literal has a clear user-facing ``description``.

Contributing back
-----------------

We invite researchers to upstream their integrations — both the integration code
and a short model page in this documentation. See the project
``CONTRIBUTING.md`` and the existing model pages under ``docs/source/models/``
(``self_forcing.rst``, ``omnidreams.rst``, ...) as templates.
