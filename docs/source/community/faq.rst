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

FAQ
===

Answers to questions that come up repeatedly in the issue tracker
and on Discord. If you don't see your question here, check
:doc:`support` for where to ask.

Getting started
---------------

What hardware do I need to run FlashDreams?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FlashDreams targets recent NVIDIA data-center GPUs. The per-method
profiling charts on each model page (e.g.
:doc:`/models/self_forcing`, :doc:`/models/lingbot_world`) run across
three devices — NVIDIA H100, GB200, and GB300.

Any CUDA-capable GPU with enough memory for the chosen checkpoint
should run the streaming methods; smaller GPUs may need to drop
multi-GPU methods back to a single device. See the
:doc:`/quickstart/index` for the cheapest path to a working clip,
and the :doc:`Models page </models/index>` for the available
methods.

Which model integrations ship in the box?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First-party model integrations ship under ``integrations/`` in the
repo. The :doc:`/models/index` page has the full list; the documented
integrations are:

- :doc:`/models/omnidreams` — HDMap-conditioned driving world model
  (the GTC 2026 closed-loop demo).
- :doc:`/models/self_forcing` — Streaming Wan 2.1 T2V (1.3B).
- :doc:`/models/causal_forcing` — Streaming Wan 2.1 T2V / I2V (1.3B).
- :doc:`/models/causal_wan22` — FastVideo Causal Wan 2.2 14B MoE T2V.
- :doc:`/models/lingbot_world` — Camera-controllable I2V world model.
- :doc:`/models/flashvsr` — Streaming video super-resolution.
- :doc:`/models/wan21` — Bidirectional Wan 2.1 T2V / I2V reference.
- :doc:`/models/cosmos_predict2` — Bidirectional Cosmos-Predict2.5
  T2V / I2V reference.

Each model page has the canonical CLI invocation, checkpoint source,
multi-GPU command, and per-method knobs.

Installation and packaging
--------------------------

Why can I install ``flashdreams`` from PyPI but not the integration packages?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only the core ``flashdreams`` package is published as a pure-Python
wheel on PyPI. Integration packages — ``flashdreams-self-forcing``,
``flashdreams-lingbot``, and the others listed in
`DEV.md <https://github.com/NVIDIA/flashdreams/blob/main/DEV.md>`__ —
are not published; they live under ``integrations/`` in the monorepo
and are designed to be consumed either as a workspace member or as
git-installable packages.

To install an integration directly from the repo:

.. code-block:: bash

   pip install "flashdreams-wan21 @ git+https://github.com/NVIDIA/flashdreams.git#subdirectory=integrations/wan21"

   # or with uv
   uv pip install "flashdreams-wan21 @ git+https://github.com/NVIDIA/flashdreams.git#subdirectory=integrations/wan21"

The rationale is in `DEV.md
<https://github.com/NVIDIA/flashdreams/blob/main/DEV.md>`__: the
core ``flashdreams`` is the only stable, pip-installable surface; the
per-integration wheels move at the upstream model's pace and stay
git-installable so they can pin against a known core commit.

Usage
-----

How do I plug in a new model integration?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`/developer_guides/new_integration` guide walks the full
flow — what to subclass on the runner side, how the entry-point
registration works, and what the per-integration directory layout
looks like. The in-tree integrations under ``integrations/``
are the canonical references; pick the one closest in shape to your
new integration and use it as a template.

The minimum surface is: subclass the right runner base, register the
slug via the ``flashdreams.runner_configs`` entry point, and (for
streaming runners) wire ``--total-blocks`` into the runner config.
:doc:`/developer_guides/new_integration` covers each step with the
exact ``pyproject.toml`` snippet.

Project and licensing
---------------------

Can I use FlashDreams commercially?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. FlashDreams is released under the
`Apache License 2.0 <https://github.com/NVIDIA/flashdreams/blob/main/LICENSE>`__,
which permits commercial use, modification, and distribution under the
license's terms. Third-party model weights and datasets used with
FlashDreams may carry their own licenses — please check those
separately.

Contributing back is welcome but not required. See
:doc:`/community/index` if you'd like to upstream a fix or improvement.

Don't see your question?
------------------------

.. container:: fd-cta-row

   .. button-link:: https://github.com/NVIDIA/flashdreams/issues
      :color: primary

      Search the issue tracker

   .. button-ref:: support
      :ref-type: doc
      :color: secondary
      :outline:

      See all support channels
