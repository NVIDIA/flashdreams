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

Evaluation
===================================

This page tracks FlashDreams performance methodology and published benchmark
artifacts. The data source is normalized JSON under ``docs/benchmarks/``.

Benchmark schema
----------------

Each benchmark record captures:

- model family and variant,
- hardware target (for example H100 / GB200 / GB300),
- workload shape and launch command metadata,
- latency breakdowns and aggregate time,
- memory metrics and pass/OOM status.

The schema and dataset are versioned in:

- ``docs/benchmarks/schema.md``
- ``docs/benchmarks/benchmark_results.json``

Latency comparison charts
-------------------------

Self-Forcing (6th block) total latency:

.. figure:: /_static/perf/self_forcing_total_ms.svg
   :class: benchmark-figure
   :alt: Self-Forcing total latency bar chart by hardware and method.

   FlashDreams vs official vs FastVideo vs LightX2V (where available).

Lingbot-World (6th block) total latency:

.. figure:: /_static/perf/lingbot_total_ms.svg
   :class: benchmark-figure
   :alt: Lingbot-World total latency bar chart by hardware and method.

   FlashDreams vs official vs FastVideo vs LightX2V (where available).

Visual quality placeholders
---------------------------

.. raw:: html

   <div class="video-slot">
     <strong>Visual Comparison Placeholder</strong><br>
     Add side-by-side YouTube videos (FlashDreams / FastVideo / LightX2V).
   </div>

Reproducibility
---------------

Charts are generated with:

.. code-block:: bash

   uv run python docs/benchmarks/generate_charts.py
