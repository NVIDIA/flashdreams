flashdreams
===================================

Overview
--------

*flashdreams* is a streaming inference pipeline for diffusion-based video
generation. It targets autoregressive ("self-forcing" / "causal-forcing")
flow-matching models with first-class support for KV-cached transformers,
ring attention across context-parallel ranks, and CUDA-graph capture for
the steady-state forward, plus a single bidirectional reference model
for parity testing.

The library is organised around a few sharp abstractions
(:doc:`apis/infra`) that every recipe (:doc:`apis/recipes`) plugs into;
shared low-level kernels and distributed helpers live under
:doc:`apis/core`. End-to-end inference scripts for every shipped model
live under ``flashdreams/examples`` and are walked through in the
sections below.

Installation
------------

The repository is a `uv <https://docs.astral.sh/uv/>`_ workspace:

.. code-block:: bash

   uv sync --extra dev
   uv run pytest flashdreams/tests

End-to-end inference scripts under ``flashdreams/examples`` additionally
require the ``examples`` extra:

.. code-block:: bash

   uv run --package flashdreams --extra examples \
       flashdreams/examples/run_alpadreams.py --help

See the project ``README.md`` for the full container-based workflow on a
Slurm node.

.. toctree::
   :maxdepth: 1
   :caption: Supported Autoregressive Models

   examples/alpadreams
   examples/self_forcing
   examples/causal_forcing
   examples/causal_wan22
   examples/lingbot_world

.. toctree::
   :maxdepth: 1
   :caption: Supported Bidirectional Models

   examples/wan21

.. toctree::
   :maxdepth: 1
   :caption: FlashDreams Inference API

   apis/core
   apis/infra
   apis/recipes

.. toctree::
   :maxdepth: 1
   :caption: FlashDreams Serving API

   apis/serving
