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
     <img class="only-light" src="_static/horizontal-light.svg" alt="FlashDreams">
     <img class="only-dark" src="_static/horizontal-dark.svg" alt="FlashDreams">
   </div>

Overview
--------------------

FlashDreams is a *high-performance inference and serving library for
interactive autoregressive video and world models*. It is a general platform
for real-time world-model applications across gaming, autonomous vehicles,
robotics, simulated or virtual environments, and more, and is the
runtime backbone of the `OmniDreams closed-loop demo at
GTC 2026 <https://research.nvidia.com/labs/sil/projects/omnidreams-blog/>`_.

.. TODO: Add the FlashDreams teaser video here once the final asset is ready.
.. .. raw:: html
..
..    <div class="video-slot">
..      <strong>Teaser video placeholder</strong><br>
..      Add the FlashDreams teaser video here.
..    </div>

.. raw:: html

   <p class="fd-subtitle">Interactive world models</p>

A world model learns to generate and evolve an environment over time. In
practice this often means video, but the same concept can include actions,
state, audio, sensor input, and control signals.

World-model serving is the runtime pattern for putting that model inside a live
application. Instead of producing one static video, the system keeps a session
alive while input, model state, GPU inference, and output evolve together. This
is useful for interactive simulation, robotics, autonomy, healthcare workflows,
creative tools, virtual worlds, and game-like experiences.

.. TODO: Vectorize this figure before final publication.
.. Figure creation trace: https://chatgpt.com/share/6a124478-4730-83e8-ba21-33628c8f1f3b
.. image:: /_static/diagrams/compare-offline-online-video-model-v2.jpg
   :alt: Offline one-shot video inference compared with online autoregressive world-model serving.

In a served world-model application, the key requirement is not only generating
a high-quality video. The runtime must keep an interactive session responsive
while the model continues to advance the world.

.. raw:: html

   <div class="fd-highlight-grid">
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Low latency</div>
       <div class="fd-highlight-body">Keep the interaction responsive when controls, sensors, or user input change.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">High throughput</div>
       <div class="fd-highlight-body">Keep the GPU busy across autoregressive steps and multi-GPU execution.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Steady streaming generation</div>
       <div class="fd-highlight-body">Stream frames or chunks at a steady pace while the session continues.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">World-state evolution</div>
       <div class="fd-highlight-body">Carry rolling state forward so the generated world evolves across steps.</div>
     </div>
   </div>

.. raw:: html

   <details class="fd-collapsible">
     <summary>Comparison with offline video generation</summary>
     <p>
       Compared with offline video generation, the target is different.
       One-shot systems prepare a conditioning input, run the model, then return a finished video.
       Libraries such as
       <a href="https://github.com/hao-ai-lab/FastVideo">FastVideo</a> and
       <a href="https://github.com/ModelTC/lightx2v">LightX2V</a>
       are strong references for high-throughput offline inference, but their
       core pattern is not a persistent interactive loop with low-latency control
       and streaming output.
     </p>
   </details>

.. raw:: html

   <details class="fd-collapsible">
     <summary>Connection to LLM serving</summary>
     <p>
       There is also a useful connection to LLM serving engines such as
       <a href="https://github.com/vllm-project/vllm">vLLM</a> and
       <a href="https://github.com/sgl-project/sglang">SGLang</a>: both LLMs and many world
       models are autoregressive. The difference is the interaction pattern.
       LLM chat usually runs <code>prefill -&gt; decode -&gt; prefill -&gt; decode</code>
       across user turns. Interactive video/world-model serving is closer to
       <code>initialize -&gt; decode -&gt; decode -&gt; decode -&gt; ...</code>:
       initialize the session once, then advance the world continuously at a fixed pace.
     </p>
   </details>

.. raw:: html

   <p class="fd-subtitle">Best-in-class inference speed</p>

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

Although FlashDreams is designed for autoregressive inference, the same
optimization stack applies naturally to bidirectional inference (e.g.,
:doc:`Wan2.1 </models/wan21>`) by treating it as a single-rollout
autoregressive pass.

.. raw:: html

   <p class="fd-subtitle">Production-oriented interactive serving backend</p>

FlashDreams also includes a production-oriented serving backend for persistent,
low-latency world-model sessions, with efficient inference execution, multi-GPU support, and
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

Start here
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
         <div class="fd-highlight-title">Model cards</div>
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
         <div class="fd-highlight-title">Developer guides</div>
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
   :caption: Model cards
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
   :caption: Developer guides
   :hidden:

   Inference pipeline overview <developer_guides/inference_pipeline_overview>
   Config system <developer_guides/config_system>
   Add a new method <developer_guides/new_integration>

.. Temporarily commented out for internal development:
..   Interactive serving architecture <developer_guides/interactive_serving>
..   Developer workflow patterns <developer_guides/usage_patterns>

.. toctree::
   :maxdepth: 2
   :caption: API and CLI
   :hidden:

   CLI reference <api/cli>
   Core <api/core>
   Infra <api/infra>
   Pipelines and runners <api/integrations>
   Serving <api/serving>

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   Contributing <community/contribute>
   Discord <community/discord>
