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

Advanced: distributed inference
===================================

FlashDreams scales out along the *context-parallel* (CP) axis: a
single recipe forward shards its sequence across ranks and recovers a
full-length attention via ring attention. This page is the operational
overview — how the launcher composes with recipe configs, what the
ranks each own, and what to read in the API reference for the
underlying primitives.

For the API surface see :doc:`../api/index`; the key entry points are
:mod:`flashdreams.core.distributed.context_parallel` and
:class:`flashdreams.core.attention.RingAttention`.

The launcher is the single source of truth
------------------------------------------

There is **no** ``cp_size`` knob on a runner config. Multi-GPU runs
are always driven by ``torchrun`` (or ``torch.distributed.run``);
the :class:`~flashdreams.infra.runner.Runner` base class initialises
``torch.distributed`` in its constructor, pins ``cuda:LOCAL_RANK``
per rank, and exposes:

- ``self.local_rank``
- ``self.global_rank``
- ``self.world_size``
- ``self.is_rank_zero``

Recipe transformers then read the CP size off the launcher's ``WORLD``
group. The same command works on 1 GPU and on 8 — only the
``--nproc_per_node`` count changes.

.. code-block:: bash

   # Single GPU.
   uv run flashdreams-run wan21-t2v-1.3b-480p

   # 4-GPU context-parallel.
   uv run torchrun --nproc_per_node=4 --no-python \
       flashdreams-run wan21-t2v-1.3b-480p

Why ``--no-python``
~~~~~~~~~~~~~~~~~~~

torchrun's default is to invoke ``python <training_script>``, which
would treat ``flashdreams-run`` as a relative path in the current
working directory and fail. ``--no-python`` tells torchrun to
``execvp`` the console-script binary directly, so PATH lookup finds
the venv shim that ``uv sync`` installed.

What ranks see
--------------

The runner is symmetrical across ranks except for I/O:

- All ranks: build the pipeline, run encoders, loop ``generate`` /
  ``finalize`` over AR blocks.
- Only rank 0: print the resolved config, log progress, write the
  ``.mp4``, dump per-step stats JSON, save any ``.pt`` outputs.

tyro's ``--help`` and parse-error banners are gated on
``LOCAL_RANK == 0`` so they print exactly once even though every rank
parses argv.

Ring attention in one paragraph
-------------------------------

Inside a context-parallel forward, each rank holds a contiguous
sequence shard of length ``S / cp_size``. The attention kernel
(:class:`flashdreams.core.attention.RingAttention`) rotates K/V
shards around the ring so every rank eventually attends against the
full sequence without ever materialising the full K/V on any single
device. The sequence-sharding helpers live in
:mod:`flashdreams.core.distributed.context_parallel` —
``split_inputs_cp`` (and ``split_inputs_cp_object_list`` for the
non-tensor variant) handle the per-rank slicing.

.. admonition:: PLACEHOLDER -- ring-attention diagram
   :class: placeholder

   **What goes here:** a small schematic of 4 ranks rotating K/V
   shards across the ring, plus the resulting per-rank Q/K attention
   block coverage.

   **Format:** SVG under ``_static/tutorials/``.

   **Source / coordinate with:** site-designer; the same image can
   anchor the API page's ring-attention section.

Constraints to know about
-------------------------

A few sharp edges to be aware of when picking ``--nproc_per_node``:

- **Sequence length must be evenly divisible by ``cp_size``.** The
  shard helper asserts this on entry. Concrete example: Wan 2.1's
  ``len_t=21`` latent window divides cleanly by 1, 3, 7, or 21 but
  not by 2, 4, or 8 — so CP must be a divisor of the recipe's
  sequence length. Recipes with power-of-two latent windows (e.g.
  template variants shipping ``len_t=8``) are happiest at CP=2/4/8.
- **Per-rank VRAM still has to fit one shard plus activations.**
  Scaling out reduces the attention activation footprint roughly
  linearly but does not shrink parameter memory; weights are
  replicated, not sharded.
- **Bandwidth matters.** Ring attention is throughput-bound on the
  K/V rotation step; NVLink-class interconnect is assumed for
  multi-GPU scaling at the published numbers.

.. admonition:: PLACEHOLDER -- scaling-curve numbers
   :class: placeholder

   **What goes here:** a small table or line chart of steady-state
   step time vs ``cp_size`` for one representative recipe (e.g.
   ``causal-wan21-causal-forcing-framewise-t2v``), so a reader can
   gauge expected scaling before booking 8 GPUs.

   **Format:** comparison table (§2.6) on the benchmarks page, linked
   from here.

   **Source / coordinate with:** benchmarks page owner.

Interaction with CUDA graphs
----------------------------

Graph capture and context parallelism compose: the captured forward
includes the ring attention's NCCL ops, so the K/V rotation is part
of the replayed graph. The standard caveats apply — drain Inductor
autotunes on the eager path before the wrapper switches into capture
mode, and call ``reset`` if you rebuild the cache or the process
group.

See :doc:`advanced-cuda-graphs` for the capture mechanics in
isolation.

Next
----

- :doc:`../api/index` — full surface for
  :mod:`flashdreams.core.distributed` and
  :mod:`flashdreams.core.attention`.
- :doc:`../examples/alpadreams` — a recipe that ships both
  single-view single-GPU and multi-view 4-GPU runners; useful as a
  worked example.
- :doc:`advanced-custom-recipe` — how to opt your own recipe into the
  same CP machinery.
