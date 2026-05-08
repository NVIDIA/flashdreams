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

Bidirectional Wan2.1, driven by ``flashdreams/examples/run_wan21.py``.
The single entry point picks T2V (1.3B) when ``--image_path`` is omitted
and I2V (14B 480P) when it is provided. Reference:
`Wan2.1 official repo <https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#run-text-to-video-generation>`_.

T2V (1.3B)
----------

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run --package flashdreams --extra examples \
     flashdreams/examples/run_wan21.py \
       --height 480 --width 832

I2V (14B 480P)
--------------

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     flashdreams/examples/run_wan21.py \
       --height 480 --width 832 \
       --image_path assets/example_data/i2v/image.jpg \
       --prompt_or_txt_path assets/example_data/i2v/prompt.txt

Run with the example data shipped in the upstream Wan2.1 repo:

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     flashdreams/examples/run_wan21.py \
       --image_path ../Wan2.1/examples/i2v_input.JPG \
       --prompt_or_txt_path "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard..."
