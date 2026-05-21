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

Adding a new recipe
===================================

flashdreams is designed for researchers to plug new streaming-inference
recipes into the existing chassis without forking the core. A *recipe*
bundles a :class:`~flashdreams.infra.diffusion.transformer.Transformer`,
its encoders / decoder, and a
:class:`~flashdreams.infra.pipeline.StreamInferencePipelineConfig`.
A *runner* wraps one such pipeline with the I/O fields (prompt, image,
output paths, …) that an end-user wants to override on the
``flashdreams-run`` command line.

Our vision is for users to keep their custom recipe in its **own
repository** that depends on ``flashdreams``, then register the
runner with the unified CLI via a Python entry point. If a custom
piece is broadly useful, we welcome a PR upstreaming it.

The :mod:`flashdreams.recipes.template` package is the minimal
end-to-end reference; clone its file layout when scaffolding a new
recipe (see ``flashdreams/flashdreams/recipes/template/README.md``).

File structure
--------------

We recommend the following layout for an external recipe package::

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

Authoring the recipe
--------------------

1. **Pipeline config.** Compose a
   :class:`~flashdreams.infra.pipeline.StreamInferencePipelineConfig`
   literal from your transformer / encoder / decoder configs. Use
   :func:`~flashdreams.infra.config.derive_config` to spawn variants
   without copy-pasting fields. ``recipe_name`` is the registry key.

2. **Runner subclass + RunnerConfig dataclass.** In ``runner.py``,
   subclass :class:`~flashdreams.infra.runner.RunnerConfig` with the
   I/O fields the CLI should expose (prompt, image path, …) and
   subclass :class:`~flashdreams.infra.runner.Runner` to implement
   :meth:`~flashdreams.infra.runner.Runner.run`: resolve runtime
   inputs, call ``self.pipeline.initialize_cache(...)``, loop
   ``generate`` + ``finalize``, then persist the output on rank 0.
   Mirror :class:`flashdreams.recipes.template.runner.TemplateRunner`
   for the canonical control flow.

3. **Per-slug runner literals.** In ``config.py``, instantiate one
   :class:`RunnerConfig` literal per shipped variant alongside the
   matching pipeline configs. ``runner_name`` is the
   ``flashdreams-run`` subcommand slug; by convention it mirrors the
   wrapped pipeline's ``recipe_name``. Always set ``description`` —
   it shows up in ``flashdreams-run --help``. These literals are the
   targets the entry-point declarations (next section) point at.

4. **Module-level dict.** Still in ``config.py``, expose a single
   ``MY_RECIPE_RUNNERS: dict[str, RunnerConfig]`` keyed by
   ``runner_name`` for programmatic use.

A minimal sketch:

.. code-block:: python

   # my_recipe/runner.py
   from dataclasses import dataclass, field

   from flashdreams.infra.runner import Runner, RunnerConfig


   @dataclass(kw_only=True)
   class MyRecipeRunnerConfig(RunnerConfig):
       """Runner config for the ``my-recipe`` family."""

       _target: type = field(default_factory=lambda: MyRecipeRunner)

       prompt: str = "A cat surfing."
       """User-overridable text prompt."""

       num_ar_steps: int = 1


   class MyRecipeRunner(Runner[MyRecipeRunnerConfig, "MyRecipePipeline"]):
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

   # my_recipe/config.py
   from flashdreams.infra.runner import RunnerConfig
   from my_recipe.runner import MyRecipeRunnerConfig

   MY_RECIPE_OFFLINE = ...   # the pipeline-config literal

   MY_RECIPE_OFFLINE_RUNNER = MyRecipeRunnerConfig(
       runner_name="my-recipe-offline",
       description="My recipe: offline reference rollout.",
       pipeline=MY_RECIPE_OFFLINE,
   )

   MY_RECIPE_RUNNERS: dict[str, RunnerConfig] = {
       cfg.runner_name: cfg for cfg in (MY_RECIPE_OFFLINE_RUNNER,)
   }

Worked end-to-end examples live in this repo at
``integrations/self_forcing/`` (Self-Forcing distilled Wan 2.1) and
``integrations/causal_forcing/`` (Causal-Forcing chunkwise /
framewise Wan 2.1). They share the same Wan 2.1 1.3B chassis but ship
as two separate plugin repos -- a useful template both for "one
external recipe per repo" and for the case where one author releases
several closely-related recipe families as independent packages.

Registering the runner with ``flashdreams-run``
-----------------------------------------------

flashdreams discovers external runners through a Python *entry point*
under the ``flashdreams.runner_configs`` group (matches nerfstudio's
``nerfstudio.method_configs`` naming). The discovery layer lives in
:mod:`flashdreams.plugins.registry`.

Add the entry point to your package's ``pyproject.toml``:

.. code-block:: toml

   [project]
   name = "my-recipe"
   dependencies = [
       "flashdreams",  # consider pinning a version, e.g. "flashdreams==X.Y.Z"
   ]

   [tool.setuptools.packages.find]
   include = ["my_recipe*"]

   [project.entry-points."flashdreams.runner_configs"]
   my-recipe-offline = "my_recipe.config:MY_RECIPE_OFFLINE_RUNNER"

You can register either a :class:`RunnerConfig` instance directly, or
a zero-arg callable that returns one (handy when construction has side
effects you want to defer until CLI time).

Install the package and the new runner appears in the CLI:

.. code-block:: bash

   pip install -e .
   flashdreams-run --help                          # lists my-recipe-offline
   flashdreams-run my-recipe-offline --help        # shows overridable fields
   flashdreams-run my-recipe-offline --prompt "..."

Built-in runners always win over a same-slug plugin: an external
package cannot silently shadow a shipped recipe.
:func:`flashdreams.configs.runner_configs.all_runners` layers
plugin-discovered runners on top of the in-tree registry returned by
:func:`flashdreams.configs.registry.supported_runners` via
:func:`~flashdreams.configs.registry.register_runner` with
``source="plugin"``, which logs and skips any slug already present.

Environment-variable backdoor
-----------------------------

When iterating on a recipe you don't always want to ``pip install`` it.
Set ``FLASHDREAMS_RUNNER_CONFIGS`` to a comma-separated list of
``slug=module.path:attribute`` pairs and the CLI picks them up at
startup:

.. code-block:: bash

   export FLASHDREAMS_RUNNER_CONFIGS="my-recipe-offline=my_recipe.config:MY_RECIPE_OFFLINE_RUNNER"
   flashdreams-run my-recipe-offline --prompt "..."

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

   flashdreams-run my-recipe-offline --prompt "A cat surfing."

Multi-GPU via context-parallelism — recipe transformers auto-detect the
CP world size from the launcher. ``--no-python`` tells ``torchrun`` to
``execvp`` the console script directly instead of wrapping it in
``python <script>``:

.. code-block:: bash

   torchrun --nproc_per_node=N --no-python flashdreams-run my-recipe-offline ...

Resolve and inspect the config without running the pipeline:

.. code-block:: bash

   flashdreams-run my-recipe-offline --no-instantiate

Programmatic access
-------------------

A recipe that hasn't been wrapped into a runner is still reachable via
its per-recipe imports — useful for serving, tests, and notebooks:

.. code-block:: python

   from my_recipe.config import MY_RECIPE_CONFIGS

   pipeline_cfg = MY_RECIPE_CONFIGS["my-recipe-offline"]
   pipeline = pipeline_cfg.setup().to("cuda")

Runners are opt-in: only register one when you want a CLI surface.

Adding a recipe to the in-tree distribution
-------------------------------------------

If your recipe lives inside this repository (under
``flashdreams/flashdreams/recipes/<name>/``), skip the entry point.
In-tree recipes self-register against the process-global registry at
import time -- the same
:func:`~flashdreams.configs.registry.register_runner` primitive the
plugin layer uses, just with ``source="builtin"`` instead of
``source="plugin"``:

1. Author ``recipes/<name>/runner.py`` with the :class:`Runner`
   subclass and its :class:`RunnerConfig` dataclass, exactly as for
   an external plugin.
2. In ``recipes/<name>/config.py``, define one :class:`RunnerConfig`
   literal per shipped variant (each with a non-empty
   ``description``) alongside the pipeline configs, collect them
   into ``<NAME>_RUNNERS``, and end the file with a tiny
   self-registration loop:

   .. code-block:: python

      from flashdreams.configs.registry import register_runner

      for _name, _cfg in MY_RECIPE_RUNNERS.items():
          register_runner(_name, _cfg, source="builtin")

   ``source="builtin"`` makes a slug collision a hard ``ValueError``
   at import time, which catches typos before the CLI even draws
   its help. **Do not do this in an out-of-tree plugin** -- plugins
   land in the live registry through the entry-point discovery layer
   in :func:`flashdreams.plugins.registry.discover_runners`, which
   passes ``source="plugin"`` automatically.
3. Add a one-line ``import flashdreams.recipes.<name>.config`` in
   ``flashdreams/configs/runner_configs.py`` so the self-registration
   side effect actually fires when the CLI starts up. The smoke test
   in ``tests/test_recipe_configs.py`` enforces parity between the
   per-recipe ``<NAME>_RUNNERS`` dicts and the live registry.

Contributing back
-----------------

We invite researchers to upstream their recipes — both the recipe code
and a short ``examples/`` page in this documentation. See the project
``CONTRIBUTING.md`` and the existing ``examples/`` pages
(``self_forcing.rst``, ``omnidreams.rst``, …) as templates.
