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

Self-Forcing
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://self-forcing.github.io/" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://arxiv.org/abs/2506.08009" target="_blank" rel="noopener noreferrer">arXiv paper</a>
     <a class="model-link-button" href="https://github.com/guandeh17/Self-Forcing" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Self-Forcing here is a Wan2.1-based text-to-video (T2V) model.
It uses a training paradigm for autoregressive video diffusion that simulates
inference-time rollout during training with KV caching, reducing the train-test
gap and enabling efficient streaming generation quality.

.. image:: https://self-forcing.github.io/static/teaser.jpg
   :alt: Self-Forcing teaser figure.
   :width: 100%

Run this model
--------------

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash --total-blocks 7

.. figure:: /_static/perf/self_forcing_total_ms.svg
   :class: benchmark-figure
   :alt: Self-Forcing total latency bar chart by hardware and method.

   DiT runtime at 6-th autoregressive rollout on a signle GPU.
