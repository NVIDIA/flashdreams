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

Get Started
===========

This page covers the path from a fresh checkout of the repository to a
generated clip on a single CUDA-capable GPU.

Install
-------

FlashDreams uses the ``uv`` python package manager. Installation
instructions for ``uv`` are available in the `Astral documentation
<https://docs.astral.sh/uv/getting-started/installation/>`_.

With ``uv`` installed, clone the repository and synchronise the workspace
environment:

.. code-block:: bash

   git clone https://github.com/NVIDIA/flashdreams.git
   cd flashdreams
   uv sync --extra dev --extra runners

The unified runner CLI is then available through ``uv run``:

.. code-block:: bash

   uv run flashdreams-run --help

Library-only install
~~~~~~~~~~~~~~~~~~~~

For projects that consume FlashDreams as a dependency rather than
running the shipped recipes, install from PyPI:

.. code-block:: bash

   pip install flashdreams

Or the current ``main`` branch:

.. code-block:: bash

   pip install "git+https://github.com/NVIDIA/flashdreams.git"

Hugging Face authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most checkpoints download from Hugging Face on first run. Export an
access token before launching:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>
   export HF_HOME=~/.cache/huggingface  # optional cache location override

For more environment and container details, see the project
`README <https://github.com/NVIDIA/flashdreams/blob/main/README.md>`_.

Speeding up CUDA builds
~~~~~~~~~~~~~~~~~~~~~~~

The first synchronisation compiles CUDA extensions from source, which
can be slow. Restricting compilation to the local GPU architecture and
parallelising build jobs reduces wall time substantially:

.. code-block:: bash

   CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
   export NVTE_CUDA_ARCHS="${CUDA_ARCH}"
   export BLOCK_SPARSE_ATTN_CUDA_ARCHS="${CUDA_ARCH}"
   export MAX_JOBS=8

The `Contributing Guide
<https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md#speeding-up-local-builds>`_
documents each variable in full and recommends an ``.envrc`` setup.

Run your first model
--------------------

Two reference paths cover the two primary usage modes: offline long
rollouts and interactive serving.

Offline inference with Self-Forcing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch an offline streaming run against the
:doc:`Self-Forcing </models/self_forcing>` Wan 2.1 1.3B T2V recipe with
the TAEHV decoder:

.. code-block:: bash

   uv run --project integrations/self_forcing \
       flashdreams-run self-forcing-wan2.1-t2v-1.3b-taehv \
       --total-blocks 7

The first invocation downloads checkpoints from Hugging Face into
``HF_HOME``. First runs take some time (Triton autotuning +
CUDA-graph warmup) but subsequent runs reuse the local cache and finish
far sooner. Output lands at
``outputs/self-forcing-wan2.1-t2v-1.3b-taehv.mp4`` (16 FPS, 480×832 by
default). See :doc:`/models/self_forcing` for ``--total-blocks``,
measured runtimes, and multi-GPU guidance.

Interactive serving with LingBot-World
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Launch the :doc:`LingBot-World </models/lingbot_world>` camera-controlled
I2V recipe with the bundled example data:

.. code-block:: bash

   uv run --project integrations/lingbot \
       flashdreams-run lingbot-world-fast \
       --example-data True \
       --total-blocks 21

Where to next
-------------

- :doc:`/models/index` — every shipped recipe with its CLI slug,
  checkpoint source, and per-recipe knobs, alongside the benchmark
  numbers.
- :doc:`/models/omnidreams` — drive a world model in real time with the
  ``interactive-drive`` demo.
- :doc:`/developer_guides/inference_pipeline_overview` — the generation
  loop end to end: KV cache, ring attention, CUDA-graph capture.
- :doc:`/developer_guides/config_system` — the configuration layer
  every recipe shares.
- :doc:`/developer_guides/new_integration` — adding a new model or
  method as a plugin.
- :doc:`/api/index` — Python API and CLI reference.
- :doc:`/models/index` — steady-state per-step latency numbers
  with reproducer commands.
- :doc:`/troubleshooting` — common first-run failures and fixes.

Project and support
-------------------

- `GitHub repository <https://github.com/NVIDIA/flashdreams>`_
- `Issues <https://github.com/NVIDIA/flashdreams/issues>`_
- `Pull requests <https://github.com/NVIDIA/flashdreams/pulls>`_
