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

Answers to questions that come up repeatedly in issues, Discord, and
Discussions. If you don't see your question here, check :doc:`support`
for where to ask.

Getting started
---------------

What hardware do I need to run FlashDreams?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: PLACEHOLDER — supported hardware answer
   :class: placeholder

   **What goes here:** A short statement of the minimum and
   recommended GPU configurations for inference and for training /
   fine-tuning, with pointers into the :doc:`benchmarks
   </benchmarks/index>` section for concrete numbers. Mention which
   compute capabilities are tested.


Which model recipes ship in the box?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: PLACEHOLDER — recipe list
   :class: placeholder

   **What goes here:** A concise list with one-line descriptions of
   each shipped recipe (causal-forcing, self-forcing, wan21,
   alpadreams, lingbot-world, fastvideo causal wan22, …) and a link
   into the corresponding tutorial or example page. Keep in sync with
   the *Supported Autoregressive Models* and *Supported Bidirectional
   Models* toctrees on the landing page.


Installation and packaging
--------------------------

Why can I install ``flashdreams`` from PyPI but not the integration packages?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: PLACEHOLDER — packaging answer
   :class: placeholder

   **What goes here:** A short explanation that only the core
   ``flashdreams`` package is published as a wheel; integrations are
   git-installable from the monorepo (with a ``pip install
   "flashdreams-<recipe> @ git+https://github.com/NVIDIA/flashdreams.git#subdirectory=integrations/<recipe>"``
   example). Cross-link to ``DEV.md`` for the rationale.


Usage
-----

How do I plug in a new model recipe?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: PLACEHOLDER — new-recipe pointer
   :class: placeholder

   **What goes here:** A one-paragraph orientation followed by a
   ``:doc:`` link to the developer guide at
   ``../developer_guides/new_recipes``. Mention the smallest set of
   abstractions a recipe must implement to plug into
   ``flashdreams-run``.


Project and licensing
---------------------

Is FlashDreams the same project as Cosmos / NeMo / NIM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. admonition:: PLACEHOLDER — relationship to other NVIDIA stacks
   :class: placeholder

   **What goes here:** A short, honest answer describing how
   FlashDreams relates to (or doesn't relate to) other NVIDIA video /
   generative stacks. If there are integration points or shared
   components, name them and link out.


Can I use FlashDreams commercially?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. FlashDreams is released under the
`Apache License 2.0 <https://github.com/NVIDIA/flashdreams/blob/main/LICENSE>`__,
which permits commercial use, modification, and distribution under the
license's terms. Third-party model weights and datasets used with
FlashDreams may carry their own licenses — please check those
separately.

Contributing back is welcome but not required. See
:doc:`contributing` if you'd like to upstream a fix or improvement.

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
