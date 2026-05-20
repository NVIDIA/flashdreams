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

Advanced: plug in your own recipe
===================================

FlashDreams is designed so that a custom streaming-inference recipe
lives in its **own package**, depends on ``flashdreams``, and registers
itself with the unified ``flashdreams-run`` CLI through a Python entry
point. You should not have to fork the core repo to ship a new recipe.

This page is the orientation. The detailed, file-by-file authoring
guide is :doc:`../developer_guides/new_recipes` — read this first to
decide which pieces apply to your model, then follow that guide for
the actual scaffolding.

When to write a recipe
----------------------

You probably want a new recipe (not just an override) if any of these
are true:

- You have a new checkpoint family whose transformer architecture
  doesn't fit any shipped recipe's
  :class:`~flashdreams.infra.diffusion.transformer.Transformer`
  subclass.
- You need a new control encoder (HDMap, camera trajectories,
  segmentation masks, …) that the shipped encoders don't model.
- You need a different streaming control flow — for example a
  rolling-forcing window that doesn't match the standard chunk-wise
  or frame-wise AR loop.

If you just need different defaults (a longer prompt, a different
scheduler step count, a different ``len_t``) you don't need a new
recipe at all — every nested field of a shipped runner is already a
CLI flag. See :doc:`your-first-generation` for the override syntax.

The two pieces you author
-------------------------

Every shipped recipe boils down to:

1. A **pipeline** —
   :class:`~flashdreams.infra.pipeline.StreamInferencePipelineConfig`
   plus the transformer, encoders, scheduler, and decoder it
   composes. This is the streaming inference "model" in the
   library's sense.
2. A **runner** —
   :class:`~flashdreams.infra.runner.RunnerConfig` plus its
   :class:`~flashdreams.infra.runner.Runner` subclass. This is the
   thin I/O layer that turns CLI flags into pipeline invocations and
   writes outputs.

The slug a user types after ``flashdreams-run`` is the
``runner_name`` on a registered ``RunnerConfig`` literal. By
convention it mirrors the wrapped pipeline's ``recipe_name``.

The fastest path to a working recipe is to copy
:mod:`flashdreams.recipes.template` — the minimal end-to-end
reference at ``flashdreams/flashdreams/recipes/template/`` — and
replace pieces one at a time. The template ships three slugs you can
exercise immediately:

- ``template-offline`` — single-step bidirectional rollout.
- ``template-autoregressive`` — streaming AR with a sliding-window KV
  cache.
- ``template-autoregressive-compiled`` — the AR variant with
  ``torch.compile`` and :class:`~flashdreams.infra.cuda_graph.CUDAGraphWrapper`
  enabled.

All three are defined in
``flashdreams/flashdreams/recipes/template/config.py``. The runner
implementation lives next door at
``flashdreams/flashdreams/recipes/template/runner.py``.

Registering with ``flashdreams-run``
------------------------------------

There are two registration paths:

- **In-tree.** Add an ``import flashdreams.recipes.<name>.config``
  line to :mod:`flashdreams.configs.runner_configs`; the recipe's
  ``register_runner`` calls run as a side effect at CLI startup.
- **Out-of-tree (recommended for external code).** Ship a Python
  entry point under the ``flashdreams.runner_configs`` group. The
  CLI's plugin layer (see
  :func:`flashdreams.plugins.registry.discover_runners`) walks
  installed entry points in that group and folds them into the runner
  registry. Built-in slugs always win over a same-slug plugin.

The entry-point group name is the load-bearing detail. Drop the
following block in your package's ``pyproject.toml``, one line per
slug you want to expose:

.. code-block:: toml

   [project.entry-points."flashdreams.runner_configs"]
   "my-recipe-t2v-720p" = "my_recipe.config:RUNNER_T2V_720P"

The left side is the entry-point name (informational only — useful
for log messages on collision); the right side is
``module:attribute``, where the attribute is a :class:`RunnerConfig`
instance (or a zero-arg factory returning one). The actual registry
key always comes from ``cfg.runner_name``.

In-repo references for the out-of-tree pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several shipped integrations live at ``integrations/<name>/`` and
register through this same mechanism. They're the cleanest copy-paste
references:

- ``integrations/self_forcing/`` — Self-Forcing distilled streaming
  T2V; entry-point declaration at
  ``integrations/self_forcing/pyproject.toml`` lines 43-45.
- ``integrations/causal_forcing/`` — Causal-Forcing's T2V and I2V
  variants; entry-point declaration at
  ``integrations/causal_forcing/pyproject.toml`` lines 44-47.

.. admonition:: In-repo workspace members vs. external packages
   :class: fd-callout

   The integrations above are workspace members
   (``[tool.uv.sources] flashdreams = { workspace = true }``), so a
   top-level ``uv sync`` installs them automatically alongside the
   core package. If you're authoring a recipe in a **separate
   repository**, the equivalent install is ``uv pip install -e
   path/to/your/recipe`` against the same venv that has
   ``flashdreams`` installed. Either way, the discovery layer just
   walks installed entry points — it has no opinion on where the
   package came from.

Either path makes the new slugs visible to ``flashdreams-run --help``
the next time the CLI starts.

What you inherit for free
-------------------------

Once a recipe is plugged in, it composes with the rest of the
library without any extra work:

- The unified CLI ergonomics — every nested config field becomes a
  ``--flag``, ``--no-instantiate`` for resolve-only debugging, tyro's
  rank-aware ``--help`` suppression on non-rank-0 ranks.
- The context-parallel launcher contract — ``torchrun`` +
  ``--no-python`` picks up your runner the same way it picks up the
  in-tree ones (see :doc:`advanced-distributed`).
- CUDA-graph capture for streaming AR — opt your transformer into
  :class:`flashdreams.infra.cuda_graph.CUDAGraphWrapper` and you get
  steady-state replay (see :doc:`advanced-cuda-graphs`).
- The smoke-test parity check in
  ``flashdreams/tests/test_recipe_configs.py``, which asserts that
  each registry key equals its config's ``runner_name`` (see
  ``test_supported_runners_keys_match_runner_name`` at lines 39-48)
  and that ``runner_name`` mirrors ``pipeline.recipe_name`` (lines
  79-92). Out-of-tree plugin recipes carry their own equivalent
  smoke tests; the in-tree aggregator-coverage check is gated on
  ``TEMPLATE_RUNNERS`` only.

Read next
---------

- :doc:`../developer_guides/new_recipes` — the long-form,
  file-by-file authoring guide. **Start here once you have decided to
  write a recipe.**
- :doc:`../examples/causal_forcing` and
  :doc:`../examples/self_forcing` — the two cleanest out-of-tree
  plugin examples shipped in this repo.
- :doc:`../api/index` — the autodoc surface for
  :class:`~flashdreams.infra.runner.RunnerConfig`,
  :class:`~flashdreams.infra.pipeline.StreamInferencePipelineConfig`,
  and the rest of the infra layer your recipe will plug into.
