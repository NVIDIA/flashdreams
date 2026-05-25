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

FlashDreams
===================================

.. raw:: html

   <style>
     #furo-main-content > section > h1 { display: none; }
   </style>
   <div class="homepage-logo-wrap">
     <img src="_static/flashdreams-logo-horizontal.png" alt="FlashDreams">
   </div>

.. .. raw:: html

..    <p style="text-align: center;"><strong>High-performance inference and serving for interactive autoregressive world models.</strong></p>

Overview
--------------------

FlashDreams is a *high-performance inference and serving library for
interactive autoregressive video and world models*. It began as the optimized
runtime behind
the `OmniDreams closed-loop demo for GTC 2026
<https://research.nvidia.com/labs/sil/projects/omnidreams-blog/>`_, and has
since grown into a general platform for realtime world-model applications across
gaming, autonomous vehicles, robotics, simulated or virtual environments, and more.

.. Best-in-class Inference Speed
.. -----------------------------

.. raw:: html

   <p class="fd-subtitle">Best-in-class inference speed.</p>

FlashDreams is engineered with efficiency in mind. With a bottom-up system
design tailored to autoregressive world-model inference patterns, it delivers best-in-class
speed across many popular open-source models and GPU architectures.

.. raw:: html

   <div class="fd-highlight-grid fd-kpi-grid">
     <a class="fd-highlight-link" href="models/self_forcing.html#profiling-benchmark">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title fd-kpi-value">2.12x</div>
         <div class="fd-highlight-body">Self-Forcing speedup</div>
       </div>
     </a>
     <a class="fd-highlight-link" href="models/lingbot_world.html#profiling-benchmark">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title fd-kpi-value">3.10x</div>
         <div class="fd-highlight-body">LingBot-World speedup</div>
       </div>
     </a>
     <a class="fd-highlight-link" href="models/wan21.html#profiling-benchmark">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title fd-kpi-value">1.40x</div>
         <div class="fd-highlight-body">Wan2.1 speedup</div>
       </div>
     </a>
     <a class="fd-highlight-link" href="models/index.html">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title fd-kpi-value">8</div>
         <div class="fd-highlight-body">Integrated models</div>
       </div>
     </a>
   </div>

.. raw:: html

   <p style="margin-top:-14px; font-size:0.82rem; color:var(--color-foreground-secondary);"><em>Although FlashDreams is designed for autoregressive inference, the same optimization stack applies naturally to bidirectional inference (e.g., Wan2.1) by treating it as a single-rollout autoregressive pass.</em></p>

.. Interactive Serving Backend
.. ---------------------------

.. raw:: html

   <p class="fd-subtitle">Production-oriented interactive serving backend.</p>

FlashDreams also includes a production-oriented serving backend for persistent,
low-latency world-model sessions, with efficient inference execution, mult-GPU support, and
streaming input/output. Explore the interactive demos powered by FlashDreams:

.. raw:: html

   <div class="fd-highlight-grid">
     <a class="fd-highlight-link" href="models/lingbot_world.html">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title">LingBot-World</div>
         <div class="fd-highlight-body">Camera-control world-model exploration.</div>
       </div>
     </a>
     <a class="fd-highlight-link" href="models/omnidreams.html">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title">OmniDreams</div>
         <div class="fd-highlight-body">Closed-loop autonomous-vehicle simulator.</div>
       </div>
     </a>
   </div>

Start Here
----------

.. raw:: html

   <div class="fd-highlight-grid">
     <a class="fd-highlight-link" href="quickstart/index.html">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title">Quickstart</div>
         <div class="fd-highlight-body">Install FlashDreams, launch your first world-model server, and start exploring quickly.</div>
       </div>
     </a>
    <a class="fd-highlight-link" href="models/index.html">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title">Model Cards</div>
         <div class="fd-highlight-body">See supported models, how to launch each one, and their performance analysis.</div>
       </div>
     </a>
     <a class="fd-highlight-link" href="api/index.html">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title">API</div>
         <div class="fd-highlight-body">Find CLI and Python API references, with links to lower-level modules.</div>
       </div>
     </a>
    <a class="fd-highlight-link" href="developer_guides/index.html">
       <div class="fd-highlight-card">
         <div class="fd-highlight-title">Developer Guides</div>
         <div class="fd-highlight-body">Learn the system design, how to integrate new models, and how to use it in your own projects.</div>
       </div>
     </a>
   </div>

.. toctree::
   :maxdepth: 1
   :caption: Quickstart
   :hidden:

   Installation <quickstart/installation>
   Launch your first model <quickstart/first_world_model>

.. toctree::
   :maxdepth: 1
   :caption: Model Cards
   :hidden:

   Self-Forcing <models/self_forcing>
   OmniDreams <models/omnidreams>
   LingBot-World <models/lingbot_world>
   Causal-Forcing <models/causal_forcing>
   Causal Wan2.2 <models/causal_wan22>
   FlashVSR <models/flashvsr>
   Wan2.1 <models/wan21>
   Cosmos-Predict2.5 <models/cosmos_predict2>

.. toctree::
   :maxdepth: 1
   :caption: Developer Guides
   :hidden:

   Offline vs online world-model flow <developer_guides/offline_vs_online>
   Runtime system overview <developer_guides/system_overview>
   Interactive serving architecture <developer_guides/interactive_serving>
   Developer workflow patterns <developer_guides/usage_patterns>
   Developer guides overview <developer_guides/index>
   Add a new model integration <developer_guides/new_recipes>
   Configure runs and overrides <developer_guides/configs>

.. toctree::
   :maxdepth: 2
   :caption: API and CLIs
   :hidden:

   CLI reference <api/cli>
   Core <api/core>
   Infra <api/infra>
   Pipelines and runners <api/recipes>
   Serving <api/serving>
