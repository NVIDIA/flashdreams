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

Serving
===================================

Serving in FlashDreams is integration-driven at the model/runtime layer, while
WebRTC provides the shared browser-facing transport for realtime demos.

Serving building blocks
-----------------------

- **Runner config** defines serving-relevant I/O fields (prompts, control
  tensors, image paths, output transport).
- **Pipeline** manages model lifecycle and cached state across steps.
- **Shared WebRTC transport** handles browser session I/O, request routing,
  data-channel controls, and media responses for realtime demos.
- **Integration runtime** owns model-specific checkpoint setup, conditioning,
  prompt/scene semantics, and chunk generation.

Reference integration
---------------------

The public WebRTC demos provide concrete examples of the shared serving stack:

- shared transport code under ``flashdreams/flashdreams/serving/webrtc/``,
- LingBot runner and runtime wiring under ``integrations/lingbot/lingbot/``,
- OmniDreams WebRTC runtime wiring under ``integrations/omnidreams/omnidreams/webrtc/``.

Launch patterns
---------------

Single GPU:

.. code-block:: bash

   uv run flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21

Multi GPU:

.. code-block:: bash

   uv run torchrun --nproc_per_node=2 --no-python flashdreams-run \
       lingbot-world-fast --example-data True --total-blocks 21

See also
--------

.. - :doc:`/developer_guides/interactive_serving`

- :doc:`/models/lingbot_world`
