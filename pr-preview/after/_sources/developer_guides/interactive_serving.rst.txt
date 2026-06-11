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

:orphan:

Interactive serving
===================================

FlashDreams serving keeps a world-model session alive while inputs and outputs
stream through the application loop.

Serving models
--------------

.. raw:: html

   <div class="fd-highlight-grid">
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Live input</div>
       <div class="fd-highlight-body">Application controls or sensor updates arrive continuously.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Warm session</div>
       <div class="fd-highlight-body">Pipeline and cache state persist across updates.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Model step</div>
       <div class="fd-highlight-body">Encoder, transformer, scheduler, and decoder advance the world.</div>
     </div>
     <div class="fd-highlight-card">
       <div class="fd-highlight-title">Streamed output</div>
       <div class="fd-highlight-body">Frames or latent output return without closing the session.</div>
     </div>
   </div>


Reference integrations
----------------------

- :doc:`/models/lingbot_world` is the primary camera-control serving reference.
- :doc:`/models/omnidreams` shows closed-loop autonomous-vehicle simulation.
- :doc:`/quickstart/first_world_model` provides the shortest command-level
  path for trying inference and serving side by side.

Serving implementation references
---------------------------------

- :doc:`/api/serving` for serving API concepts and component mapping.
- :doc:`/developer_guides/inference_pipeline_overview` for runner/pipeline
  execution flow.
- ``integrations/lingbot/lingbot/webrtc`` for the WebRTC serving stack.
