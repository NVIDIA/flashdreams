AlpaDreams
===================================

Driving-scene video generation with the Alpadreams recipe (Cosmos DiT +
HDMap conditioning + I2V mask injection). Driver:
``flashdreams/examples/run_alpadreams.py``. Checkpoints and example
data are auto-downloaded on first run.

The launcher picks one of :data:`ALPADREAMS_CONFIG_BUILDERS` based on
``--n_cameras``:

- ``--n_cameras 1`` — single front-facing camera, defaults to
  ``sv_2steps_chunk2_loc6_lightvae_lighttae``.
- ``--n_cameras 4`` — four surrounding cameras, defaults to
  ``mv_2steps_chunk4_loc8_pshuffle_lighttae``.

Single GPU, single view
-----------------------

.. code-block:: bash

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
       flashdreams/examples/run_alpadreams.py \
       --n_cameras 1 --total_blocks 20

Add ``--overwrite_config_name sv_2steps_chunk2_loc6_lightvae_lighttae_perf``
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
