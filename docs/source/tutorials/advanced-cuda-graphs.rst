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

Advanced: CUDA graphs
===================================

FlashDreams' streaming AR transformers run the steady-state forward
inside a captured ``torch.cuda.CUDAGraph``: after a short warmup, the
same fixed-shape forward replays from a pre-recorded graph so we pay
launch overhead once, not per step. This page is a tour of when that
applies, how to toggle it, and what to watch out for.

For the API surface — flags, dataclasses, methods — see the
:doc:`API reference <../api/index>`. The implementation lives in
:class:`flashdreams.infra.cuda_graph.CUDAGraphWrapper`.

When CUDA graph capture applies
-------------------------------

The wrapper captures a callable that:

- runs entirely on CUDA;
- has stable input shapes across calls (a steady AR step over a
  fixed-size window cache);
- has stable parameter storage (no allocator churn between calls);
- mutates its internal buffers in place (e.g. KV-cache slots) rather
  than reallocating them.

The streaming AR recipes are designed around exactly that contract:
the KV-cache window is fixed at pipeline init, every AR step is the
same shape, and the cache slots are updated in place. So once the
first rollout has "drained" Inductor's lazy autotunes on the eager
path, every subsequent step is a graph replay.

Bidirectional reference recipes (e.g. ``wan21-t2v-1.3b-480p``) do not
benefit from graph capture in the same way — there is one big forward
pass, not a steady-state loop — so they leave the wrapper disabled.

.. admonition:: PLACEHOLDER -- before/after CUDA-graph profile
   :class: placeholder

   **What goes here:** a side-by-side Nsight Systems or
   ``torch.profiler`` screenshot showing the launch-overhead delta on
   a streaming AR run, before vs. after graph capture takes over.

   **Format:** PNG under ``_static/tutorials/``.

   **Source / coordinate with:** benchmarks page; the same trace
   should back the headline "steady-state step" stat.

How to toggle it
----------------

Each recipe's pipeline config wires the graph wrapper through a
``use_cuda_graph`` flag on its compiled transformer config. The
in-tree template recipe ships three variants — the
``template-autoregressive-compiled`` slug
(``TEMPLATE_AUTOREGRESSIVE_COMPILED`` at
``flashdreams/flashdreams/recipes/template/config.py:170``) is the one
that sets ``use_cuda_graph=True``; ``template-offline`` and
``template-autoregressive`` leave it disabled. After warmup (default
``warmup_iters=2`` per
:class:`flashdreams.infra.cuda_graph.CUDAGraphWrapper`), the wrapper
flips into capture mode and subsequent calls replay the graph.

To turn graph capture off on a per-run basis without editing code,
override the field at the CLI. The exact override path depends on the
runner, but it always resolves through the transformer config:

.. code-block:: bash

   uv run flashdreams-run <slug> \
       --pipeline.diffusion-model.transformer.use-cuda-graph False

.. admonition:: PLACEHOLDER -- confirmed override flag per runner
   :class: placeholder

   **What goes here:** a small list of the actual CLI flag path on
   each shipped streaming AR runner (alpadreams, self-forcing,
   causal-forcing, fastvideo-causal-wan22, lingbot-world). The nested
   config schema is uniform, but the leaf flag name should be
   confirmed by running ``flashdreams-run <slug> --help | grep cuda``
   so we don't ship typos.

   **Format:** definition-list or bullet list of slug -> flag.

   **Source / coordinate with:** docs build; verifiable at CI time.

What can break capture
----------------------

The most common failure modes — and what they look like — when graph
capture goes wrong:

- **Triton/Inductor autotune during capture.** A ``torch.compile``\ d
  function will JIT and autotune kernels on its first call per shape;
  doing that inside the captured stream raises
  ``cudaErrorStreamCaptureUnsupported``. ``CUDAGraphWrapper.drain`` is
  provided so recipes can run a first eager rollout to drain those
  autotunes before the wrapper switches into capture mode.
- **Signature drift.** Passing a new positional-tensor or
  kwarg-tensor signature drops the captured graph and restarts
  warmup. This is intentional, but it surprises people who toggle
  optional inputs mid-rollout.
- **External state swap.** If the caller swaps the cache out from
  under the graph (a fresh streaming session, a different rollout),
  call :meth:`~flashdreams.infra.cuda_graph.CUDAGraphWrapper.reset` so
  the wrapper drops and re-captures against the new buffers.

.. admonition:: PLACEHOLDER -- worked walk-through of a capture failure
   :class: placeholder

   **What goes here:** a real terminal session showing a
   ``cudaErrorStreamCaptureUnsupported`` raised by a missing drain,
   with the resolving change highlighted.

   **Format:** fenced ``console`` code-block (~25 lines).

   **Source / coordinate with:** capture from CI on a known-bad
   commit; the failure mode is deterministic.

Interaction with ``torch.compile``
----------------------------------

The streaming AR recipes pair the CUDA-graph wrapper with
``torch.compile`` in the ``max-autotune-no-cudagraphs`` mode (see
:mod:`flashdreams.infra.compile`). The "no-cudagraphs" suffix is
intentional — Inductor's own graph-capture pass is disabled because
the wrapper does the capture explicitly, on the steady-state forward,
where the AR control flow is already settled.

Mixing the two requires that the eager-path "drain" step run all
Inductor autotunes before the wrapper trips into capture mode; the
template recipe demonstrates the standard pattern.

Next
----

- :doc:`advanced-distributed` — how context-parallel ranks interact
  with the same captured forward.
- :doc:`../api/index` — :class:`CUDAGraphWrapper` reference and the
  surrounding ``flashdreams.infra`` surface.
