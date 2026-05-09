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

AlpaDreams
===================================

Driving-scene video generation with the Alpadreams recipe (Cosmos DiT +
HDMap conditioning + I2V mask injection). Driver:
``flashdreams/examples/run_alpadreams.py``. Checkpoints and example
data are auto-downloaded on first run.

The launcher picks one of :data:`ALPADREAMS_CONFIGS` based on
``--n_cameras``:

- ``--n_cameras 1`` — single front-facing camera, defaults to
  ``alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae``.
- ``--n_cameras 4`` — four surrounding cameras, defaults to
  ``alpadreams-mv-2steps-chunk4-loc8-pshuffle-lighttae``.

Single GPU, single view
-----------------------

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
       flashdreams/examples/run_alpadreams.py \
       --n_cameras 1 --total_blocks 20

Add ``--overwrite_config_name alpadreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf``
for the perf-tuned variant (CUDA-graph captured forward + light VAE/TAE).

Multi GPU, multi view
---------------------

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
       flashdreams/examples/run_alpadreams.py \
       --n_cameras 4 --total_blocks 20

Each rank owns one camera; ring attention shards the per-camera context
across the world.

Diffusion forcing, single view
------------------------------

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
       flashdreams/examples/run_alpadreams.py \
       --n_cameras 1 \
       --total_blocks 12 \
       --overwrite_config_name alpadreams-sv-35steps-chunk2-loc24-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m \
       --offload_text_encoder

With the usual ``--total_blocks 12`` rollout, the chunk2 checkpoint decodes to
93 frames.

Bidirectional, single view
--------------------------

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
       flashdreams/examples/run_alpadreams.py \
       --n_cameras 1 \
       --total_blocks 1 \
       --num_chunks 24 \
       --overwrite_config_name alpadreams-sv-35steps-chunk48-loc48-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m \
       --offload_text_encoder

The bidirectional checkpoint generates one full block per run. Omit
``--num_chunks`` for the trained 48-chunk length, or set ``--num_chunks 24`` for
a shorter 93-frame run.

Credentials
-----------

Checkpoints are pulled from the team S3 bucket. Drop a JSON file at
``credentials/s3_checkpoint.secret`` with ``aws_access_key_id``,
``aws_secret_access_key``, ``endpoint_url``, ``region_name`` and the
loader picks it up automatically.

A HuggingFace token is also required for the encoder weights:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>
   export HF_HOME=~/.cache/huggingface              # optional
   export FLASHDREAMS_CACHE_DIR=~/.cache/flashdreams # optional
