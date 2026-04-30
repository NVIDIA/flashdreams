Causal Wan2.2 (FastVideo)
===================================

FastVideo-style Wan2.2 causal T2V, driven by
``flashdreams/examples/run_causal_wan22.py``. Reference:
`FastVideo basic_self_forcing_causal_wan2_2_i2v.py <https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_self_forcing_causal_wan2_2_i2v.py>`_.

T2V only for now: the FastVideo Wan2.2 checkpoint's I2V protocol
(one-shot first-frame VAE-seed warmup) does not fit the unified
streaming pipeline's per-AR-step mask-injection I2V and is not wired
here.

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   uv run --package flashdreams --extra examples \
     python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
       flashdreams/examples/run_causal_wan22.py \
       --total_blocks 21
