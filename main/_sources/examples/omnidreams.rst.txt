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

Omnidreams
===================================

Driving-scene video generation with the Omnidreams recipe (Cosmos DiT +
HDMap conditioning + I2V mask injection). Driver: the unified
``flashdreams-run`` CLI; checkpoints and example data are auto-downloaded
on first run.

The runner slug encodes every variant of :data:`OMNIDREAMS_CONFIGS`. Pick
the one matching the camera setup; for example:

- single view -> ``omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae``
- single view, perf preset -> ``omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf``
- 4-camera multi view -> ``omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae``

Pass ``--example-data`` to lazy-sync the bundled HDMap clips + first frames
into ``assets/example_data/omnidreams/`` and fill the per-camera path tuples.
Drop the flag and pass ``--hdmap-video-paths`` / ``--first-frame-paths``
explicitly for production runs.

Single GPU, single view (perf preset)
-------------------------------------

.. code-block:: bash

   uv run flashdreams-run \
       omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --example-data True --total-blocks 20

Multi GPU, multi view
---------------------

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       omnidreams-mv-2steps-chunk4-loc8-pshuffle-lighttae \
       --example-data True --total-blocks 20

Each rank owns one camera; ring attention shards the per-camera context
across the world.

Diffusion forcing, single view
------------------------------

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       omnidreams-sv-35steps-chunk2-loc24-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m \
       --example-data True --total-blocks 12

With the usual ``--total-blocks 12`` rollout, the chunk2 checkpoint decodes
to 93 frames.

Bidirectional, single view
--------------------------

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       omnidreams-sv-35steps-chunk48-loc48-cosmos2-2b-res720p-30fps-hdmap-vae-mads1m \
       --example-data True --total-blocks 1 \
       --pipeline.diffusion-model.transformer.len-t 24

The bidirectional checkpoint generates one full block per run. Omit the
``--pipeline.diffusion-model.transformer.len-t`` override for the
trained 48-chunk length, or set it to 24 for a shorter 93-frame run.

Encoder offload
---------------

For tight VRAM budgets, precompute the one-shot encoders on a single GPU
and reuse the embeddings on the AR pass:

.. code-block:: bash

   # Rank-0-only producer: dump text + first-frame embeddings.
   uv run flashdreams-run \
       omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --example-data True \
       --save-embeddings-path /tmp/alpa_emb.pt

   # Consumer: skip the one-shot encoders, hydrate the cache from .pt.
   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf \
       --example-data True \
       --embeddings-path /tmp/alpa_emb.pt \
       --total-blocks 20

Credentials
-----------

Omnidreams resolves public Omni Dreams checkpoints and example-data repos from
the ``nvidia`` Hugging Face org by default (``nvidia/omni-dreams-models`` and
``nvidia/omni-dreams-samples``). Set ``OMNI_DREAMS_HF_ORG`` before invoking
``flashdreams-run`` if your token can only access another mirror, such as
``nvidia-omni-dreams-lha``.

A Hugging Face token is required for the encoder weights and the selected Omni
Dreams org:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>
   export HF_HOME=~/.cache/huggingface              # optional
   export FLASHDREAMS_CACHE_DIR=~/.cache/flashdreams # optional
   # export OMNI_DREAMS_HF_ORG=nvidia-omni-dreams-lha

Internal S3-backed runs can set ``FLASHDREAMS_INTERNAL_STORAGE=1`` to switch
checkpoint and example-data URLs back to ``s3://flashdreams``. When using that
path, drop a JSON file at ``credentials/s3_checkpoint.secret`` with
``aws_access_key_id``, ``aws_secret_access_key``, ``endpoint_url``, and
``region_name``; the loader picks it up automatically.
