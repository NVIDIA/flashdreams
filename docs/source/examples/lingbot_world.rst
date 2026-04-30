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
