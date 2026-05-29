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

The documentation is organized into two surfaces. The **developer
guides** are conceptual walkthroughs of how the inference pipeline,
configuration layer, and integration surface fit together. The
**CLI and API reference** is the autodoc-driven enumeration of every
public class, function, and command-line flag.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Developer guides
      :link: /developer_guides/index
      :link-type: doc

      Conceptual walkthroughs of the inference pipeline, the shared
      configuration system, the integration surface for adding a
      method, common usage patterns, and interactive serving.

   .. grid-item-card:: CLI and API reference
      :link: /api/index
      :link-type: doc

      Autodoc surface for the ``flashdreams-run`` CLI, the core
      runtime, the infrastructure layer, the pipelines and runners,
      and the serving components.

.. toctree::
   :hidden:
   :maxdepth: 1

   Developer guides </developer_guides/index>
   CLI and API reference </api/index>
