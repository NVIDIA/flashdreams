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

Wan2.1 (bidirectional)
===================================

Bidirectional Wan2.1, driven by the unified ``flashdreams-run`` CLI. The
two shipped runners are ``wan21-t2v-1.3b-480p`` and ``wan21-i2v-14b-480p``.
Reference: `Wan2.1 official repo
<https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#run-text-to-video-generation>`_.

T2V (1.3B)
----------

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run flashdreams-run wan21-t2v-1.3b-480p

I2V (14B 480P)
--------------

``--image-path`` defaults to the bundled ``assets/example_data/i2v/image.jpg``
demo frame so the I2V runner produces a video out of the box:

.. code-block:: bash

   uv run flashdreams-run wan21-i2v-14b-480p

Run with the example data shipped in the upstream Wan2.1 repo:

.. code-block:: bash

   uv run flashdreams-run wan21-i2v-14b-480p \
       --image-path ../Wan2.1/examples/i2v_input.JPG \
       --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard..."
