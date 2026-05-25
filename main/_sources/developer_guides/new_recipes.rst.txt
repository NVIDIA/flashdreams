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

FlashDreams is built so new world/video models can plug into the runtime
without forking the core library. This page is the detailed authoring guide for
external integrations. For the repo-wide architecture, read
:doc:`/developer_guides/system_overview` first.

Use this guide when you want to:

- wrap an existing video/world model behind ``flashdreams-run``;
- build an external package that depends on ``flashdreams``;
- expose a model through both Python and CLI entry points;
- upstream a polished integration back into this repository.

Integration mental model
------------------------

An integration has two layers:

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Pipeline

      Defines generation behavior. It composes transformer, scheduler,
      optional encoder, and optional decoder configs into a
      ``StreamInferencePipelineConfig``.

   .. grid-item-card:: Runner

      Defines user-facing I/O. It turns CLI fields such as prompt, image path,
      input video, output directory, and runtime toggles into pipeline calls.

The code still uses ``recipe_name`` for the stable pipeline slug, but the
authoring model to follow is **pipeline + runner**.

.. raw:: html

   <div class="ai-figure-placeholder">
     <div class="ai-figure-title">Figure placeholder: adding a FlashDreams model integration</div>
     <div class="ai-figure-body">
       Replace this block with an AI-generated 16:9 figure that shows an
       external model repository plugging into FlashDreams through pipeline
       configs, runner configs, and the flashdreams.runner_configs entry point.
     </div>
   </div>

.. dropdown:: AI figure prompt

   .. code-block:: text

      Create a polished technical illustration for a developer guide titled
      "Adding a FlashDreams model integration".

      Show an external model repository on the left with files: runner.py,
      config.py, transformer/, optional encoder/, optional decoder.py, and
      pyproject.toml. Show arrows into the FlashDreams runtime on the right:
      RunnerConfig -> Runner -> StreamInferencePipeline -> Encoder -> Transformer
      / Scheduler -> Decoder -> Output. Also show a Python entry point named
      flashdreams.runner_configs connecting the external package to the unified
      flashdreams-run CLI.

      Make the diagram attractive and simple: modern developer documentation
      style, vector-like blocks, tasteful accent colors, clean arrows, readable
      labels, soft shadows, no clutter. Emphasize that the external integration
      remains in its own repository while reusing FlashDreams core/infra. Aspect
      ratio 16:9, high resolution, crisp text.

Recommended workflow
--------------------

1. Start from the closest existing integration under ``integrations/*``.
2. Create a standalone Python package that depends on ``flashdreams``.
3. Implement the pipeline components and config literals.
4. Implement a runner that owns CLI-facing I/O.
5. Register runner configs through the ``flashdreams.runner_configs`` entry
   point.
6. Add a model page and tests once the integration is stable.

The in-repo integrations are the reference design. Treat each folder under
``integrations/*`` as an effectively standalone plugin repository that can be
developed and released independently.

File structure
--------------

Use this layout for an external integration package::

    my_integration/
    ├── my_integration/
    │   ├── __init__.py
    │   ├── runner.py            # Runner + RunnerConfig + I/O helpers
    │   ├── config.py            # Pipeline and runner config literals
    │   ├── pipeline.py          # optional: custom pipeline/cache behavior
    │   ├── transformer/         # model network + Transformer + AR cache
    │   ├── encoder/             # optional: control, text, image, or video encoders
    │   ├── decoder.py           # optional: latent-to-pixel streaming decoder
    │   └── ...
    └── pyproject.toml

Add optional files only when the model actually needs them. Most integrations
can use the base ``StreamInferencePipeline`` directly and only provide model
components plus config literals.

Authoring checklist
-------------------

Pipeline config
^^^^^^^^^^^^^^^

Compose a :class:`~flashdreams.infra.pipeline.StreamInferencePipelineConfig`
literal from transformer, scheduler, optional encoder, and optional decoder
configs.

- Give every variant a stable ``recipe_name``. Treat it like a route, not a
  local variable.
- Use :func:`~flashdreams.infra.config.derive_config` to create variants from a
  base config instead of copy-pasting full nested literals.
- Keep configs as plain data. Put per-rollout shape checks in cache
  initialization, not in config constructors.

Runner
^^^^^^

In ``runner.py``, subclass :class:`~flashdreams.infra.runner.RunnerConfig` with
the I/O fields the CLI should expose, then subclass
:class:`~flashdreams.infra.runner.Runner` and implement
:meth:`~flashdreams.infra.runner.Runner.run`.

A runner should:

- resolve user inputs such as prompts, images, videos, or control files;
- call ``self.pipeline.initialize_cache(...)`` once per rollout/session;
- loop over ``generate`` and ``finalize`` for autoregressive steps;
- write outputs only on rank 0.

Runner configs
^^^^^^^^^^^^^^

In ``config.py``, create one runner config literal per shipped variant. Set a
clear ``description`` because it appears in ``flashdreams-run --help``. By
convention, ``runner_name`` mirrors the wrapped pipeline's ``recipe_name``.

Also expose a module-level dict keyed by ``runner_name`` for programmatic use:

.. code-block:: python

   MY_INTEGRATION_RUNNERS: dict[str, RunnerConfig] = {
       cfg.runner_name: cfg for cfg in (MY_MODEL_FAST_RUNNER,)
   }

Minimal sketch
--------------

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
               # Save out to cfg.output_dir / f"{cfg.runner_name}.<ext>"
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
``fastvideo_causal_wan22/``, ``omnidreams/``, ``flashvsr/``, and
``cosmos_predict2/``.
Each folder contains its own ``pyproject.toml``, pipeline config, and runner
definitions, and should be treated as a standalone plugin-style repository.

Browse the corresponding GitHub folders directly:

- `integrations/self_forcing <https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing>`_
- `integrations/causal_forcing <https://github.com/NVIDIA/flashdreams/tree/main/integrations/causal_forcing>`_
- `integrations/lingbot <https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot>`_
- `integrations/wan21 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/wan21>`_
- `integrations/fastvideo_causal_wan22 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/fastvideo_causal_wan22>`_
- `integrations/omnidreams <https://github.com/NVIDIA/flashdreams/tree/main/integrations/omnidreams>`_
- `integrations/flashvsr <https://github.com/NVIDIA/flashdreams/tree/main/integrations/flashvsr>`_
- `integrations/cosmos_predict2 <https://github.com/NVIDIA/flashdreams/tree/main/integrations/cosmos_predict2>`_

Registering the runner with ``flashdreams-run``
-----------------------------------------------

FlashDreams discovers external runners through a Python *entry point*
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

When iterating on an integration, you do not always want to ``pip install`` it.
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

Testing and docs
----------------

Before upstreaming or sharing the integration, add the same practical surface
users expect from built-in models:

- a model page under ``docs/source/models/`` with install and launch commands;
- smoke tests for config construction and runner registration;
- parity or profiling tests when comparing against an official implementation;
- small example inputs or clear instructions for obtaining them.

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
