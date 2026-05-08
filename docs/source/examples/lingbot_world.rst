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

Lingbot-World
===================================

Camera-controlled image-to-video with the Lingbot-World recipe.
Reference:
`lingbot-world fast inference <https://github.com/robbyant/lingbot-world?tab=readme-ov-file#fast-inference>`_.

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

Single GPU
----------

Currently, even single GPU inference requires `torchrun` to be used (in order to set the right env variables).

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     torchrun --standalone --nnodes=1 --nproc_per_node=1 \
       -m flashdreams.examples.run_lingbot_world \
       --total_blocks 21

Multi GPU
---------

Wan 2.1 context parallel assumes `cp_size == world_size`, so Lingbot World can be launched
with `torchrun` across multiple GPUs:

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     torchrun --standalone --nnodes=1 --nproc_per_node=2 \
       -m flashdreams.examples.run_lingbot_world \
       --total_blocks 21
