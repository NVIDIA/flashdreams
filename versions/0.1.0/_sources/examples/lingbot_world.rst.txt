Lingbot-World
===================================

Camera-controlled image-to-video with the Lingbot-World recipe.
Reference:
`lingbot-world fast inference <https://github.com/robbyant/lingbot-world?tab=readme-ov-file#fast-inference>`_.

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
       flashdreams/examples/run_lingbot_world.py \
       --total_blocks 21
