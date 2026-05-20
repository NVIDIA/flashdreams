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

Your first generation
===================================

This walkthrough drives the bidirectional Wan2.1 1.3B T2V runner end
to end. It is the smallest shipped runner that produces a real video,
fits on a single 24 GB GPU, and exercises every layer of the pipeline
that downstream tutorials touch.

By the end of this page you will have:

- a generated ``.mp4`` on disk;
- a printed copy of the resolved
  :class:`~flashdreams.infra.runner.RunnerConfig`;
- a working mental model of how recipe slugs, overrides, and runners
  fit together.

If you have not already, complete :doc:`quickstart` first.

Pick a runner
-------------

We use ``wan21-t2v-1.3b-480p`` — the T2V variant of the
:doc:`Wan2.1 bidirectional recipe <../examples/wan21>`. It is the
single-AR-step reference path; it generates a short clip from a text
prompt alone, no first frame required.

The slug resolves to ``RUNNER_WAN21_T2V_1PT3B_480P`` in
``integrations/wan21/wan21/config.py``. That literal is a real
:class:`~flashdreams.infra.runner.RunnerConfig`, so every field on it
(the prompt, the output path, the scheduler step count, the
transformer's ``len_t``) becomes a CLI flag.

Sanity-check the config
-----------------------

Before downloading checkpoints, confirm the slug resolves on your
install:

.. code-block:: bash

   uv run flashdreams-run wan21-t2v-1.3b-480p --no-instantiate

``--no-instantiate`` short-circuits after printing the resolved
config. You will see a ``Wan21T2VRunnerConfig`` block, including the
nested ``WanInferencePipelineConfig`` and its scheduler / transformer
defaults. Nothing here touches the GPU.

.. admonition:: PLACEHOLDER -- captured ``--no-instantiate`` output
   :class: placeholder

   **What goes here:** the first ~30 lines of the resolved config
   block from ``flashdreams-run wan21-t2v-1.3b-480p --no-instantiate``.

   **Format:** fenced ``console`` code-block.

   **Source / coordinate with:** capture against the same release the
   docs ship against; the dataclass repr is stable across patch
   versions but the precise default values may shift.

Set up Hugging Face access
--------------------------

The Wan2.1 1.3B checkpoint is fetched from Hugging Face on first
invocation. Set a token in your shell:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

   # Optional: override default cache locations.
   export HF_HOME=~/.cache/huggingface
   export FLASHDREAMS_CACHE_DIR=~/.cache/flashdreams

The first ``flashdreams-run`` call downloads the diffusion-model
``safetensors`` shard under ``$HF_HOME``; subsequent runs reuse the
cache.

Run with the bundled prompt
---------------------------

Defaults are sufficient for the first run — the runner ships a
default prompt, an output path under ``outputs/``, and a sensible
total block count. Just call the slug:

.. code-block:: bash

   uv run flashdreams-run wan21-t2v-1.3b-480p

Rank 0 logs progress; on completion the generated video is written to
``outputs/wan21-t2v-1.3b-480p.mp4`` and a per-step stats JSON is
emitted under ``outputs/`` when the recipe's
``enable_sync_and_profile`` flag is on (it is, by default, for this
runner).

.. admonition:: PLACEHOLDER -- sample generation
   :class: placeholder

   **What goes here:** a short clip or static thumbnail of the
   generated output so a reader knows what "success" looks like.

   **Format:** ``_static/tutorials/wan21-t2v-sample.mp4`` (or a poster
   JPG referenced via the §2.3 video-embed snippet).

   **Source / coordinate with:** marketing / site-designer; needs to
   be re-rendered each time the default prompt or scheduler defaults
   change.

Override the prompt
-------------------

Every nested field of the runner config is exposed as a CLI flag.
The most common override is the prompt itself:

.. code-block:: bash

   uv run flashdreams-run wan21-t2v-1.3b-480p \
       --prompt "A cat surfing on a turquoise wave at sunset."

To redirect the output to a different directory:

.. code-block:: bash

   uv run flashdreams-run wan21-t2v-1.3b-480p \
       --prompt "A cat surfing on a turquoise wave at sunset." \
       --output-dir runs/cat-surfing

The runner writes ``<output-dir>/wan21-t2v-1.3b-480p.mp4`` plus a
per-step stats file at ``<output-dir>/stats_wan21-t2v-1.3b-480p.json``
— the filename is derived from the runner slug, not configurable.

Nested config fields use a dotted path. For example, to drop the
scheduler's inference step count from 50 to 30 for a faster (lower
quality) run:

.. code-block:: bash

   uv run flashdreams-run wan21-t2v-1.3b-480p \
       --pipeline.diffusion-model.scheduler.num-inference-steps 30

Use ``flashdreams-run wan21-t2v-1.3b-480p --help`` to list every
flag the runner exposes.

Scale to multiple GPUs
----------------------

Multi-GPU is just torchrun plus the same command — the recipe
transformer auto-detects its context-parallel size from the launcher's
``WORLD`` group, and the runner gates its I/O on
``self.is_rank_zero`` so only one rank writes outputs.

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python \
       flashdreams-run wan21-t2v-1.3b-480p \
       --prompt "A cat surfing on a turquoise wave at sunset."

``--no-python`` matters here: see :doc:`advanced-distributed` for why
torchrun needs it to dispatch the ``flashdreams-run`` console script,
plus the broader distributed picture (ring attention, sharded
sequences, divisibility constraints).

What's actually happening
-------------------------

The Wan2.1 T2V runner is the simplest path through the streaming
pipeline:

1. ``flashdreams-run`` parses argv, resolves the
   ``Wan21T2VRunnerConfig`` literal, prints it on rank 0, and calls
   ``config.setup()`` to build the runner.
2. The runner constructs the pipeline (encoders, scheduler,
   transformer, decoder), pulling checkpoints from Hugging Face on
   first use.
3. ``runner.run()`` encodes the prompt once, calls
   ``pipeline.initialize_cache(...)``, then loops ``generate`` and
   ``finalize`` for each AR block — for this runner, exactly one
   block, ``len_t=21`` latent frames.
4. The streaming VAE decoder turns latents into pixels frame-by-frame
   as they leave the transformer.
5. Rank 0 writes the ``.mp4`` and the per-step stats JSON to
   ``outputs/``.

If you want to see the same control flow written out as code, the
canonical reference is ``flashdreams/recipes/template/runner.py`` —
the minimal end-to-end runner — and ``integrations/wan21/wan21/``
for the Wan2.1-specific specialisations.

Next
----

- :doc:`../examples/wan21` — full per-recipe walkthrough for Wan2.1
  T2V and I2V.
- :doc:`advanced-cuda-graphs` — how the steady-state forward gets
  captured into a CUDA graph and when it applies.
- :doc:`advanced-distributed` — context-parallel ranks, ring
  attention, and the multi-GPU launcher.
- :doc:`advanced-custom-recipe` — pointer to the full guide for
  scaffolding your own recipe.
