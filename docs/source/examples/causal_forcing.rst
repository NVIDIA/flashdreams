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

The causal-forcing variants of Wan2.1 swap in the
``causal_forcing_framewise`` config on the same launcher
(``flashdreams/examples/run_causal_wan21.py``). Whether the run is T2V
or I2V is decided by the presence of ``--image_path``.

T2V
---

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
       flashdreams/examples/run_causal_wan21.py \
       --total_blocks 21 \
       --overwrite_config_name causal_forcing_framewise

I2V
---

Pass an image plus the matching prompt; the driver wires them through
the per-AR-step mask-injection I2V path:

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
       flashdreams/examples/run_causal_wan21.py \
       --total_blocks 21 \
       --overwrite_config_name causal_forcing_framewise \
       --prompt_or_txt_path assets/example_data/i2v/prompt.txt \
       --image_path assets/example_data/i2v/image.jpg
