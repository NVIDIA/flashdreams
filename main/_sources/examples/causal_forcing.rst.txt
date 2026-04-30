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
