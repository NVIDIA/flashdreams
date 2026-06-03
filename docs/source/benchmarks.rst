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

Benchmarks
==========

.. container:: fd-hero fd-hero-band

   .. rubric:: Benchmarks
      :class: fd-hero-title

   .. container:: fd-hero-lede

      Per-step latency at steady state for the recipes we profile,
      measured on H100, GB200, and GB300, and compared against an
      existing implementation of the same model on the same hardware
      and checkpoint.

   .. container:: fd-cta-row

      .. button-ref:: /quickstart/index
         :ref-type: doc
         :color: primary

         Reproduce locally

      .. button-link:: https://github.com/NVIDIA/flashdreams
         :color: secondary
         :outline:

         GitHub

Results — autoregressive
------------------------

Per-AR-step latency at steady state, across the three devices we
profile against. Each cell is the median of the AR-2-onward window
(warm-up steps dropped, steady-state CUDA graph captured) of a single
run on that device. Lower is better.

.. container:: fd-compare fd-compare-numeric

   .. list-table::
      :header-rows: 1
      :widths: 36 16 16 16 16

      * - Recipe (runner slug)
        - H100 (ms)
        - GB200 (ms)
        - GB300 (ms)
        - Source
      * - ``self-forcing-wan2.1-t2v-1.3b-taehv``
        - 344
        - 203
        - 171
        - ``_static/performance/self_forcing/perf-0521.md``
      * - ``lingbot-world-fast`` (4×GPU, Ulysses)
        - 629
        - 449
        - 394
        - ``_static/performance/lingbot_world/perf-0521.md``

Results — bidirectional
-----------------------

Bidirectional recipes are treated as a single large-windowed causal
rollout in FlashDreams (see each integration's README for the full
statement). Reporting per-step ms lines up the bidirectional reference
with the streaming variants above. Measured on a single GPU.

.. container:: fd-compare fd-compare-numeric

   .. list-table::
      :header-rows: 1
      :widths: 40 18 18 18

      * - Recipe (runner slug)
        - H100 (ms)
        - GB200 (ms)
        - GB300 (ms)
      * - ``wan21-t2v-1.3b-480p``
        - 1040
        - 441
        - 382

.. admonition:: Per-device profile charts
   :class: fd-callout

   The interactive per-device profiling charts on each model page
   (e.g. :doc:`/models/self_forcing`, :doc:`/models/lingbot_world`,
   :doc:`/models/wan21`) read from the same per-recipe markdown files
   under ``_static/performance/`` as the tables above.

Results — super-resolution
--------------------------

FlashVSR is a streaming video super-resolution recipe, so its latency
scales with output resolution rather than autoregressive step count and
is reported per scene. Measured on GB200, FlashDreams against the
official FlashVSR runner; source
``_static/performance/flashvsr/perf-0527.md``. Lower is better.

.. container:: fd-compare fd-compare-numeric

   .. list-table::
      :header-rows: 1
      :widths: 34 33 33

      * - Scene (resolution)
        - FlashDreams (ms)
        - Official (ms)
      * - 384×384
        - 74.7
        - 130.5
      * - 672×384
        - 114.5
        - 235.7
      * - 384×672
        - 111.2
        - 181.3
      * - 640×480
        - 132.5
        - 255.7
      * - 768×416
        - 135.5
        - 215.9
      * - 1280×704
        - 372.5
        - 528.2

Versus upstream
---------------

The FlashDreams runner calls the same model code paths as the upstream
library it integrates with, but in a different inference environment:
KV caches managed by ``flashdreams.infra``, ring attention provided by
``flashdreams.core``, and a CUDA graph captured per recipe. For an
apples-to-apples comparison, both sides are forced to the cuDNN
attention backend under matched runtime settings. Lower is better; a
ratio greater than 1 means FlashDreams is faster.

.. container:: fd-compare fd-compare-numeric

   .. list-table::
      :header-rows: 1
      :widths: 30 10 16 16 12 16

      * - Recipe (runner slug)
        - GPU
        - Upstream (ms)
        - FlashDreams (ms)
        - Ratio
        - Baseline
      * - ``self-forcing-wan2.1-t2v-1.3b-taehv``
        - H100
        - 432
        - 344
        - 1.26×
        - Official Self-Forcing runner
      * - ``self-forcing-wan2.1-t2v-1.3b-taehv``
        - GB200
        - 350
        - 203
        - 1.72×
        - Official Self-Forcing runner
      * - ``self-forcing-wan2.1-t2v-1.3b-taehv``
        - GB300
        - 251
        - 171
        - 1.47×
        - Official Self-Forcing runner
      * - ``self-forcing-wan2.1-t2v-1.3b-taehv``
        - GB300
        - 362
        - 171
        - 2.12×
        - FastVideo runner (landing-page hero)
      * - ``lingbot-world-fast`` (4×GPU)
        - H100
        - 1950
        - 629
        - 3.10×
        - Official LingBot-World runner
      * - ``lingbot-world-fast`` (4×GPU)
        - GB200
        - 1113
        - 449
        - 2.48×
        - Official LingBot-World runner
      * - ``wan21-t2v-1.3b-480p``
        - GB300
        - 534
        - 382
        - 1.40×
        - FastVideo runner (landing-page hero)
      * - ``wan21-t2v-1.3b-480p``
        - H100
        - 1290
        - 1040
        - 1.24×
        - FastVideo runner

Supported models
----------------

The benchmarks above cover the recipes currently in the profiled
corpus. Streaming and autoregressive recipes emit per-AR-step output
and target sub-second steady-state step latency once the CUDA graph is
captured; bidirectional recipes emit one end-to-end output per
invocation and serve as the parity reference for the streaming
variants. Each tile links to the recipe's page, which carries the
canonical invocation, the checkpoint source, and the per-recipe knobs.

.. container:: fd-eyebrow

   Streaming and autoregressive

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Self-Forcing
      :class-card: fd-feature
      :link: /models/self_forcing
      :link-type: doc

      Streaming Wan 2.1 T2V via the Self-Forcing plugin. AR steps
      after warmup are sub-second on H100 / GB200.

   .. grid-item-card:: Causal-Forcing
      :class-card: fd-feature
      :link: /models/causal_forcing
      :link-type: doc

      Causal-forcing framewise T2V and I2V variants of Wan 2.1 via
      the Causal-Forcing plugin.

   .. grid-item-card:: Causal Wan 2.2
      :class-card: fd-feature
      :link: /models/causal_wan22
      :link-type: doc

      FastVideo Wan 2.2 14B causal T2V recipe.

   .. grid-item-card:: LingBot-World
      :class-card: fd-feature
      :link: /models/lingbot_world
      :link-type: doc

      Camera-controlled I2V with bundled prompt, first-frame, and
      camera arrays.

   .. grid-item-card:: OmniDreams
      :class-card: fd-feature
      :link: /models/omnidreams
      :link-type: doc

      Single-view and multi-view streaming recipes against the
      OmniDreams checkpoints, including a diffusion-forcing AR
      variant.

   .. grid-item-card:: FlashVSR
      :class-card: fd-feature
      :link: /models/flashvsr
      :link-type: doc

      Streaming video super-resolution for the FlashVSR checkpoint
      family.

.. container:: fd-eyebrow

   Bidirectional reference

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Wan 2.1
      :class-card: fd-feature
      :link: /models/wan21
      :link-type: doc

      Bidirectional Wan 2.1 — T2V 1.3B / 480p and I2V 14B / 480p.
      The parity baseline for ``self-forcing`` and
      ``causal-forcing`` recipes.

   .. grid-item-card:: Cosmos-Predict2.5
      :class-card: fd-feature
      :link: /models/cosmos_predict2
      :link-type: doc

      Bidirectional Cosmos-Predict2 recipes (T2V / I2V, 2B).

What we measure
---------------

The reported metric is **steady-state per-step latency**: the
wall-clock of one autoregressive step once past AR step 2 and the
steady-state CUDA graph is captured. This is the ``total(w/o
finalize)`` value the inference pipeline logs each step. Time to first
frame is not reported, and quality metrics (FVD, CLIP-T) are out of
scope here — they are tracked by each recipe's training pipeline. The
benchmark only verifies that the inference path holds tolerance-bounded
parity with the upstream reference, which is enforced per recipe (see
*A note on parity* below).

Methodology
-----------

Each row is produced by driving a recipe end-to-end and parsing the
per-step log lines: drop the first two AR steps as warm-up, then take
the median of the remaining ``total(w/o finalize)`` values.

.. code-block:: bash

   uv run flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b-taehv \
       --total-blocks 7 \
       2>&1 | tee /tmp/bench-self-forcing.log

``--total-blocks`` is defined on the streaming-runner subclasses
(``self_forcing``, ``causal_forcing``, ``fastvideo_causal_wan22``,
``lingbot``, ``omnidreams``); bidirectional runners (``wan21-*``,
``cosmos2-*``) drop the flag and emit a single end-to-end output.

For multi-GPU recipes, launch the same command under ``torchrun``; the
recipe transformer auto-detects its context-parallel size from the
launcher's world group (LingBot-World uses Ulysses sequence
parallelism across 4 GPUs):

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python \
       flashdreams-run lingbot-world-fast --total-blocks 21

The upstream baseline in *Versus upstream* runs the same checkpoint
under the upstream library's own runner; per-integration instructions
live under ``integrations/<name>/tests/parity_check/``.

.. admonition:: A note on parity
   :class: fd-callout

   Six integrations ship a parity check against their upstream
   reference under ``integrations/<name>/tests/parity_check/run.sh``:
   ``self_forcing``, ``lingbot``, ``wan21``, ``cosmos_predict2``,
   ``flashvsr``, and ``hy_worldplay`` (intended for manual execution
   in the upstream environment, not in CI). The numbers on this page
   assume parity holds where it is enforced; see
   :doc:`/community/index` for how to escalate a regression rather
   than averaging away a discrepancy.

Related
-------

- The :doc:`/developer_guides/index` cover the architecture behind the
  recipes you can run today.
- :doc:`/community/index` lists the channels to use if a number on
  this page does not reproduce on your hardware.

.. toctree::
   :hidden:
   :maxdepth: 1

   /models/self_forcing
   /models/causal_forcing
   /models/causal_wan22
   /models/cosmos_predict2
   /models/flashvsr
   /models/lingbot_world
   /models/omnidreams
   /models/wan21
