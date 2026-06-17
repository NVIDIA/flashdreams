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

Documentation
=============

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Developer guides
      :link: /developer_guides/index
      :link-type: doc

      Guides that teach you how to understand, extend, and build on
      FlashDreams, combining conceptual explanation with worked code
      examples.

   .. grid-item-card:: CLI and API Reference
      :link: /api/index
      :link-type: doc

      Complete reference for the ``flashdreams-run`` CLI, the core
      runtime, the infrastructure layer, the pipelines and runners,
      and the serving components.

   .. grid-item-card:: Troubleshooting
      :link: /troubleshooting
      :link-type: doc

      Common first-run failures (e.g. CUDA build mismatches, disk and
      cache limits, Hugging Face authentication, GPU memory)
      each with the likely cause and the next step to try.

Where to start
--------------

New to FlashDreams? Start with :doc:`Get Started </quickstart/index>` to go
from a fresh checkout to a running model. From there, read the
:doc:`developer guides </developer_guides/index>` for the conceptual model,
reach for the :doc:`CLI and API reference </api/index>` when you need exact
symbols and flags, and check :doc:`/troubleshooting` when a first run fails.

.. toctree::
   :hidden:
   :maxdepth: 1

   /developer_guides/index
   /api/index
   /troubleshooting
