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

.. Revised by @ruilongl on 2026-05-23. Future agent changes can ignore this file.

What is world model serving?
============================

A world model learns to generate and evolve an environment over time. Today
that often means video, but the same idea can include actions, state, audio,
sensor input, and control signals.

World model serving is the runtime pattern for putting that model inside a live
application. Instead of producing one static clip, the system keeps a session
alive while input, model state, GPU inference, and output evolve together. This
is useful for interactive simulation, robotics, autonomy, healthcare workflows,
creative tools, virtual worlds, and gaming-like experiences.

.. Figure creation trace: https://chatgpt.com/share/6a124478-4730-83e8-ba21-33628c8f1f3b

.. image:: /_static/diagrams/compare-offline-online-video-model-v2.jpg
   :alt: Offline one-shot video inference compared with online autoregressive world-model serving.

In a served world-model application, the important requirement is not just
generating a good clip. The runtime must keep an interactive session responsive
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
       <div class="fd-highlight-title">Streaming generation at steady pace</div>
       <div class="fd-highlight-body">Stream frames or chunks at a steady pace while the session continues.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">World state evolvement</div>
       <div class="fd-highlight-body">Carry rolling state forward so the generated world evolves across steps.</div>
     </div>
   </div>

Compared with offline video generation, the target is different.
One-shot systems prepare a conditioning input, run the model, then return a
finished video. Libraries such as
`FastVideo <https://github.com/hao-ai-lab/FastVideo>`_ and
`LightX2V <https://github.com/ModelTC/lightx2v>`_ are strong references for
high-throughput offline video inference, but their core pattern is not a
persistent interactive loop with low-latency control and streaming output.

There is also a useful connection to LLM serving engines such as
`vLLM <https://github.com/vllm-project/vllm>`_ and
`SGLang <https://github.com/sgl-project/sglang>`_: both LLMs and many world
models are autoregressive. The difference is the interaction pattern. LLM chat
usually runs ``prefill -> decode -> prefill -> decode`` across user turns.
Interactive video/world-model serving is closer to
``initialize -> decode -> decode -> decode -> ...``: initialize the session
once, then advance the world continuously at a fixed pace.

FlashDreams uses this online serving path for interactive integrations such as
:doc:`/models/lingbot_world` and :doc:`/models/omnidreams`. To try a model,
start with :doc:`/quickstart/first_world_model`.
