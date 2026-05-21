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

Install in 60 seconds
===================================

This page is the shortest path from a fresh clone to a working
``flashdreams-run`` CLI. You will need:

- An x86_64 Linux host with at least one CUDA-capable GPU. Any recipe
  that downloads a real checkpoint will want substantially more VRAM
  than a laptop GPU — :doc:`your-first-generation` calls out concrete
  requirements per slug.
- `uv <https://docs.astral.sh/uv/>`__ installed on PATH. Everything
  else (Python interpreter, PyTorch, FlashDreams packages and every
  in-tree integration) is pulled into a project-local virtual
  environment by ``uv sync``.
- A Hugging Face token if you intend to pull gated checkpoints. Set
  ``HF_TOKEN`` in your shell before running any recipe that downloads
  weights.

Clone and sync
--------------

.. code-block:: bash

   git clone https://github.com/NVIDIA/flashdreams.git
   cd flashdreams

   # Workspace install: core + every integration + dev tooling.
   uv sync --extra dev --group lint

The ``--extra dev`` group pulls test and lint dependencies; the
``--group lint`` adds pre-commit so ``uv run pre-commit run -a`` works
without a second sync. See ``DEV.md`` for the full development
workflow (linting, type checking, IDE setup).

Add the runner extras
---------------------

The ``flashdreams-run`` CLI lazy-imports ``mediapy`` and ``opencv`` for
image and video I/O. Install the ``runners`` extra the first time you
intend to actually generate a video, not just resolve a config:

.. code-block:: bash

   uv sync --extra dev --extra runners --group lint

You can skip this if you only want to inspect what a runner would do —
see the ``--no-instantiate`` flag in :doc:`your-first-generation`.

Verify the CLI
--------------

``flashdreams-run`` dispatches over a tyro subcommand union built from
the runner registry; ``--help`` lists every registered slug along with
its one-line description.

.. code-block:: bash

   uv run flashdreams-run --help

.. admonition:: PLACEHOLDER -- captured ``flashdreams-run --help`` output
   :class: placeholder

   **What goes here:** the literal first ~40 lines of
   ``uv run flashdreams-run --help`` so a reader can scan the available
   runner slugs without running the command themselves.

   **Format:** fenced ``console`` code-block.

   **Source / coordinate with:** capture from a CI-built image so the
   slug list matches the release. Until then we rely on the live
   command being the source of truth.

The slug list is the source of truth for what ``flashdreams-run`` can
do — every entry corresponds to a
:class:`~flashdreams.infra.runner.RunnerConfig` literal registered by
an in-tree recipe or an installed plugin. Two slugs you will use
repeatedly:

- ``wan21-t2v-1.3b-480p`` — bidirectional Wan2.1 T2V, the smallest
  shipped runner that produces a real video; used in
  :doc:`your-first-generation`.
- ``template-offline`` — the reference template recipe; useful for
  smoke-testing the CLI and inspecting resolved configs without
  touching a GPU.

Per-runner help
~~~~~~~~~~~~~~~

Every overridable field on the resolved config is auto-exposed as a
CLI flag. Pass ``--help`` after a slug to see them:

.. code-block:: bash

   uv run flashdreams-run wan21-t2v-1.3b-480p --help

The output is generated dynamically by tyro from the runner config
dataclass, so flags such as ``--prompt``, ``--output-dir``,
``--pixel-height``, and the nested
``--pipeline.diffusion-model.transformer.len-t`` are all present
without any extra glue. Streaming AR runners (e.g.
``self-forcing-wan2.1-t2v-1.3b-flash``) additionally expose
``--total-blocks``; the bidirectional ``wan21-*`` and non-streaming
``cosmos2-*`` runners do not.

Resolve a config without a GPU
------------------------------

The ``--no-instantiate`` flag prints the resolved config and exits
without constructing the runner. It is the fastest way to confirm
your install is healthy and your overrides parsed the way you
expected:

.. code-block:: bash

   uv run flashdreams-run template-offline --no-instantiate

You should see the resolved ``RunnerConfig`` printed once on the
local-rank-0 process, then a clean exit. If this fails you are
looking at a packaging or environment problem, not a model issue.

Troubleshooting
---------------

- **``flashdreams-run: command not found``** — make sure you are
  invoking it through ``uv run`` so the project virtualenv is on
  PATH, or activate the venv at ``.venv/bin/activate``.
- **A recipe slug is missing from ``--help``** — out-of-tree plugins
  (Self-Forcing, Causal-Forcing, etc.) ship as workspace-member
  packages under ``integrations/``. ``uv sync`` installs them
  automatically because they're declared as workspace members in the
  root ``pyproject.toml``; if a slug is still missing, check that the
  integration's ``pyproject.toml`` declares the ``flashdreams.runner_configs``
  entry-point group correctly (see
  :doc:`advanced-custom-recipe` for the wiring).
- **``HF_TOKEN`` not set** — checkpoint downloads from Hugging Face
  silently fail or 401 for gated repos. Export ``HF_TOKEN`` before
  invoking the runner; ``HF_HOME`` and ``FLASHDREAMS_CACHE_DIR``
  default to ``~/.cache/huggingface`` and ``~/.cache/flashdreams``.

Next
----

Continue to :doc:`your-first-generation` to drive
``wan21-t2v-1.3b-480p`` end to end and confirm the install really
works on a GPU.
