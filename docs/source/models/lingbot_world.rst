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

Lingbot-World
===================================

Overview
--------

Lingbot-World enables camera-controllable image-to-video generation with fast
streaming inference and context-parallel runtime support.

Links
-----

- Project page: `Lingbot-World GitHub <https://github.com/robbyant/lingbot-world>`_
- Integration package: `flashdreams/integrations/lingbot <https://github.com/NVIDIA/flashdreams/tree/main/integrations/lingbot>`_

Run this model
--------------

.. code-block:: bash

   uv run flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21

.. figure:: /_static/perf/lingbot_total_ms.svg
   :class: benchmark-figure
   :alt: Lingbot-World total latency bar chart by hardware and method.

   DiT runtime at 6-th autoregressive rollout on 4x GPUs.
