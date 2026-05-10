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
    │   ├── runner.py            # RunnerConfig literal(s) + Runner subclass
    │   ├── config.py            # StreamInferencePipelineConfig literal(s)
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

2. **Runner config.** Subclass
   :class:`~flashdreams.infra.runner.RunnerConfig`, add the I/O fields
   the CLI should expose (prompt, image path, …), and instantiate one
   literal per shipped variant. ``runner_name`` is the
   ``flashdreams-run`` subcommand slug; by convention it mirrors the
   wrapped pipeline's ``recipe_name``. Always set ``description`` —
   it shows up in ``flashdreams-run --help``.

3. **Runner subclass.** Subclass
   :class:`~flashdreams.infra.runner.Runner` and implement
   :meth:`~flashdreams.infra.runner.Runner.run`: resolve runtime inputs,
   call ``self.pipeline.initialize_cache(...)``, loop ``generate`` +
   ``finalize``, then persist the output on rank 0. Mirror
   :class:`flashdreams.recipes.template.runner.TemplateRunner` for the
   canonical control flow.

4. **Module-level dict.** Expose a single
   ``MY_RECIPE_RUNNERS: dict[str, RunnerConfig]`` keyed by
   ``runner_name``.

5. **Self-register at import time.** Each recipe ``runner.py`` calls
   :func:`flashdreams.configs.registry.register_runner` once per slug
   so the in-tree CLI picks the runner up just by importing the
   module. ``source="builtin"`` makes a slug collision a hard
   ``ValueError`` at import time, which catches typos before the CLI
   even draws its help.

A minimal sketch:

.. code-block:: python

   # my_recipe/runner.py
   from dataclasses import dataclass, field

   from flashdreams.configs.registry import register_runner
   from flashdreams.infra.runner import Runner, RunnerConfig
   from my_recipe.config import MY_RECIPE_OFFLINE


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


   MY_RECIPE_OFFLINE_RUNNER = MyRecipeRunnerConfig(
       runner_name="my-recipe-offline",
       description="My recipe: offline reference rollout.",
       pipeline=MY_RECIPE_OFFLINE,
   )

   MY_RECIPE_RUNNERS: dict[str, RunnerConfig] = {
       cfg.runner_name: cfg for cfg in (MY_RECIPE_OFFLINE_RUNNER,)
   }

   for _name, _cfg in MY_RECIPE_RUNNERS.items():
       register_runner(_name, _cfg, source="builtin")

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
   my-recipe-offline = "my_recipe.runner:MY_RECIPE_OFFLINE_RUNNER"

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

   export FLASHDREAMS_RUNNER_CONFIGS="my-recipe-offline=my_recipe.runner:MY_RECIPE_OFFLINE_RUNNER"
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
``flashdreams/flashdreams/recipes/<name>/``), skip the entry point —
the same :func:`~flashdreams.configs.registry.register_runner`
primitive the plugin layer uses also covers the in-tree case:

1. Author ``recipes/<name>/runner.py`` with one
   :class:`RunnerConfig` literal per shipped variant (each with a
   non-empty ``description``), a ``<NAME>_RUNNERS`` dict, and a loop
   that calls
   :func:`~flashdreams.configs.registry.register_runner` for each
   slug with ``source="builtin"`` (see the sketch above).
2. Add a one-line ``import flashdreams.recipes.<name>.runner`` in
   ``flashdreams/configs/runner_configs.py`` so the side effects
   actually fire when the CLI starts up. ``source="builtin"`` makes a
   slug collision a hard ``ValueError`` at import time. The smoke
   test in ``tests/test_recipe_configs.py`` enforces parity.

Contributing back
-----------------

We invite researchers to upstream their recipes — both the recipe code
and a short ``examples/`` page in this documentation. See the project
``CONTRIBUTING.md`` and the existing ``examples/`` pages
(``self_forcing.rst``, ``alpadreams.rst``, …) as templates.
