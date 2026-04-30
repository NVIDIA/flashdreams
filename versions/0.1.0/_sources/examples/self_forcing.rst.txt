Self-forcing T2V (Wan2.1)
===================================

Self-forcing T2V variant of Wan2.1, driven by
``flashdreams/examples/run_causal_wan21.py``. The default config is the
self-forcing recipe; checkpoints are auto-downloaded from HuggingFace
on first run.

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>
   export HF_HOME=~/.cache/huggingface  # optional

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
       flashdreams/examples/run_causal_wan21.py \
       --total_blocks 7
