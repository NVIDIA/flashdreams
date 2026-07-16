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

Serving in FlashDreams uses a protocol-neutral session lifecycle with
integration-provided model workers. The :doc:`serving API design
</developer_guides/serving_api>` documents the endpoint contract, worker
scheduler, multi-session behavior, and Dynamo extension point.

Serving building blocks
-----------------------

- **Serve model config** publishes model capabilities and a lazy worker factory.
- **Session service** owns session leases, sequence numbers, and worker placement.
- **Protocol transport** maps WebSocket, WebRTC, or gRPC onto the common service.
- **Model worker** owns shared weights and one or more isolated session caches.

Reference integration
---------------------

:doc:`LingBot-World </models/lingbot_world>` provides the canonical serving
integration:

- runner and pipeline wiring under ``integrations/lingbot/lingbot/``,
- interactive transport under ``integrations/lingbot/lingbot/webrtc/``.

Launch patterns
---------------

Single GPU:

.. code-block:: bash

   uv run flashdreams-serve lingbot-world-fast --protocol webrtc --eager-load

Multi-GPU workers publish ``ResourceRequest.gpu_count > 1`` and rely on a
rank-aware scheduler/worker implementation to create their process group. The
current Lingbot serving adapter advertises one GPU and one session per worker;
its existing multi-rank WebRTC bootstrap remains available through the
integration-specific launch script while that worker is migrated.

See also
--------

.. - :doc:`/developer_guides/interactive_serving`

- :doc:`/models/lingbot_world`
