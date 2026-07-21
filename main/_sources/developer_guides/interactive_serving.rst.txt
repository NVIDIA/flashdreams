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

- :doc:`/models/omnidreams` shows closed-loop autonomous-vehicle simulation.
- :doc:`/models/lingbot_world` is the primary camera-control serving reference.
- :doc:`/quickstart/index` provides the shortest command-level path for
  trying inference and serving side by side.

Serving implementation references
---------------------------------

- :doc:`/api/serving` for serving API concepts and component mapping.
- :doc:`/developer_guides/inference_pipeline_overview` for runner/pipeline
  execution flow.
- ``flashdreams.serving.webrtc`` for the shared WebRTC server, session manager,
  runtime protocol, data-channel messages, packaged UI app factory, and
  distributed serve-loop helpers.
- ``integrations/lingbot/lingbot/webrtc`` and
  ``integrations/omnidreams/omnidreams/webrtc`` for concrete WebRTC demos built
  on the shared serving stack.

WebRTC demo shape
-----------------

New realtime demos should use WebRTC as the primary browser transport. The
shared package owns the reusable serving shell:

- ``runtime.WebRTCGenerationRuntime`` describes the lifecycle a model runtime
  must provide: initialize, reset a rollout, generate one chunk, and close.
- ``manager.BaseWebRTCSessionManager`` owns one active peer connection, control
  data-channel parsing, liveness, keyboard resampling, loopback warmup, and
  chunk scheduling.
- ``server.create_packaged_webrtc_app`` builds the aiohttp app from packaged
  browser assets and keeps those resources alive until cleanup.
- ``bootstrap.initialize_cuda_distributed`` and ``bootstrap.run_webrtc_server``
  provide the common CUDA/distributed launch and teardown path.
- ``messages`` defines the common action, event, heartbeat, disconnect,
  ``chunk_done``, ``event_ack``, and error payload contracts.

Model integrations should keep model-specific runtime code local: checkpoint
setup, scene or prompt semantics, conditioning/rendering, cache math, and any
custom HTTP routes that configure a session. The shared WebRTC layer should be
enough for a new demo to provide a runtime, package its UI assets, add optional
routes, and start serving without copying another demo's bootstrap code.
