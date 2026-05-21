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

Causal-Forcing
===================================

Overview
--------

Causal-Forcing provides streaming Wan2.1 generation variants for both text-to-video
and image-to-video workflows, tuned for stable autoregressive generation.

Links
-----

- Project page: `Causal-Forcing GitHub <https://github.com/LiRunyi2001/causal-forcing>`_
- Integration package: `flashdreams/integrations/causal_forcing <https://github.com/NVIDIA/flashdreams/tree/main/integrations/causal_forcing>`_

Run this model
--------------

T2V:

.. code-block:: bash

   uv run flashdreams-run \
       causal-wan21-causal-forcing-framewise-t2v --total-blocks 21

I2V:

.. code-block:: bash

   uv run flashdreams-run \
       causal-wan21-causal-forcing-framewise-i2v --total-blocks 21 \
       --image-path assets/example_data/i2v/image.jpg \
       --prompt-path assets/example_data/i2v/prompt.txt
