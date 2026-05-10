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

Causal-forcing T2V / I2V (Wan2.1)
=================================

The causal-forcing variants of Wan2.1 are shipped as the
``flashdreams-causal-forcing`` integration plugin and exposed as
separate runners on the unified ``flashdreams-run`` CLI:

- ``causal-forcing-wan2.1-t2v-1.3b-chunkwise`` — chunkwise T2V (``len_t=3``).
- ``causal-forcing-wan2.1-t2v-1.3b-framewise`` — framewise T2V (``len_t=1``).
- ``causal-forcing-wan2.1-i2v-1.3b-framewise`` — framewise I2V (``len_t=1``).

T2V
---

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run flashdreams-run \
       causal-forcing-wan2.1-t2v-1.3b-framewise --total-blocks 21

I2V
---

The I2V runner defaults ``--image-path`` to the plugin's bundled
``integrations/causal_forcing/assets/image.jpg`` demo frame; pass an
explicit path to override:

.. code-block:: bash

   uv run flashdreams-run \
       causal-forcing-wan2.1-i2v-1.3b-framewise --total-blocks 21
