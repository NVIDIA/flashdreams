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

   .. container:: fd-hero-eyebrow

      Measurement, not marketing

   .. rubric:: How fast is FlashDreams, really?
      :class: fd-hero-title

   .. container:: fd-hero-lede

      FlashDreams is built for **steady-state streaming video
      diffusion** — the interesting numbers are not "time to first
      frame" but the cost of every subsequent autoregressive step
      once KV caches are warm, CUDA graphs are captured, and the
      pipeline is in its hot loop. This page documents what we
      measure, on what hardware, with what software, and how
      FlashDreams compares to upstream baselines.

   .. container:: fd-cta-row

      .. button-link:: https://github.com/NVIDIA/flashdreams/blob/main/PERFORMANCE.md
         :color: primary

         Read PERFORMANCE.md

      .. button-ref:: /quickstart/index
         :ref-type: doc
         :color: secondary
         :outline:

         Reproduce locally

Headline metrics
----------------

.. container:: fd-eyebrow

   Steady-state, post-warmup, post-graph-capture

.. container:: fd-lede

   Four numbers per recipe. The same grid leads the
   :doc:`landing page </index>`; figures here are the source of
   truth and the landing page mirrors them.

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            TBD

         .. container:: fd-stat-label

            Steady-state step (ms)

         .. container:: fd-stat-note

            H100, ``self-forcing-wan2.1-t2v-1.3b-flash``.

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            TBD

         .. container:: fd-stat-label

            FlashDreams / upstream

         .. container:: fd-stat-note

            Same recipe, same GPU; ratio > 1 means faster.

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            TBD

         .. container:: fd-stat-label

            Peak memory (GiB)

         .. container:: fd-stat-note

            Steady-state, post-graph-capture.

   .. grid-item::

      .. container:: fd-stat

         .. container:: fd-stat-value

            TBD

         .. container:: fd-stat-label

            Scaling at 8 GPUs

         .. container:: fd-stat-note

            Ring attention, bidirectional reference.

.. admonition:: PLACEHOLDER — headline stat values
   :class: placeholder

   **What goes here:** four ≤ 6-character stat values for the grid
   above. Pick the four that read best on the landing-page tile too.

   **Source data:** rows of the Results tables below; do not invent.

   **Reproduce with:** the per-row CLI invocations in *Methodology*.

.. admonition:: New to streaming inference?
   :class: fd-callout

   Start with :doc:`/index` for an overview, then walk a recipe in the
   :doc:`/quickstart/index` before reading these numbers in context.
   The architectural rationale lives in the :doc:`/api/index`.

What these benchmarks measure
-----------------------------

.. container:: fd-eyebrow

   Scope and metrics

.. container:: fd-lede

   Per-recipe, we report four numbers. Quality metrics (FVD, CLIP-T)
   are tracked by each recipe's training pipeline and are **out of
   scope** here — we only verify that the inference path is bit-for-
   bit (or tolerance-bounded) parity with the upstream reference.
   Parity status is noted per recipe.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Steady-state step latency
      :class-card: fd-feature

      Wall-clock of one AR step (``diffuse`` + ``decode`` +
      ``finalize``) once past AR step 2 and the CUDA graph is
      captured. p50 and p95 of the steady window.

   .. grid-item-card:: Throughput
      :class-card: fd-feature

      Frames per second once steady; for bidirectional recipes,
      total frames / total wall-clock for one full generation.

   .. grid-item-card:: Peak GPU memory
      :class-card: fd-feature

      ``torch.cuda.max_memory_allocated`` in GiB, plus reserved
      memory for fragmentation analysis.

   .. grid-item-card:: Scaling efficiency
      :class-card: fd-feature

      Multi-GPU throughput as a fraction of the ideal linear scaling
      baseline; reported with the ring-attention topology used.

Methodology
-----------

.. container:: fd-eyebrow

   Four templates, every row reproducible

.. container:: fd-split fd-split-asymmetric-reverse

   .. container:: fd-split-text

      Every Results row is generated by one of the four invocations
      below. We do not report mean — single-step JIT / allocator
      outliers skew it. All step-latency numbers are the **median over
      five repeat runs** of the AR-2-onward window. p95 is reported
      alongside p50 only when the spread is informative (> 5 % of p50).

      **1. Single-GPU per-step latency.** Drive a streaming recipe
      end-to-end, parse the per-step log lines, drop the first two AR
      steps as warm-up, then take the median and 95th percentile of
      the remaining ``total(w/o finalize)`` values.

      .. code-block:: bash

         uv run flashdreams-run \
             self-forcing-wan2.1-t2v-1.3b-flash \
             --total-blocks 7 \
             2>&1 | tee /tmp/bench-self-forcing.log

      **2. Throughput.** Same invocation; take steady-state window as
      ``total − warmup`` and divide generated frame count by it.

      .. code-block:: bash

         uv run flashdreams-run \
             <streaming-recipe-slug> --total-blocks <N> \
             --output-dir /tmp/bench-frames

      ``--total-blocks`` is defined on streaming-runner subclasses
      (``self_forcing``, ``causal_forcing``,
      ``fastvideo_causal_wan22``, ``lingbot``, ``omnidreams``);
      bidirectional and non-streaming runners (``wan21-*``,
      ``cosmos2-*``) drop the flag and emit a single end-to-end
      output.

      **3. Multi-GPU / ring-attention scaling.** Launch with
      ``torchrun``; the recipe transformer auto-detects its
      context-parallel size from ``torchrun``'s ``WORLD`` group (no
      ``--ring-size`` flag — the launcher is the source of truth).
      Sweep world size 1 → 2 → 4 → 8.

      .. code-block:: bash

         uv run torchrun --nproc_per_node=8 --no-python \
             flashdreams-run wan21-t2v-1.3b-480p

      **4. Upstream-baseline parity.** Same recipe, same model
      checkpoint, under the upstream library's own environment. Feeds
      the *Versus upstream* section.

      .. code-block:: bash

         # See PERFORMANCE.md for the upstream env setup.
         PYTHONPATH=./flashdreams python -m flashdreams.scripts.cli \
             self-forcing-wan2.1-t2v-1.3b-flash --total-blocks 7

   .. container:: fd-split-visual

      .. container:: fd-info-card

         .. container:: fd-info-card-title

            What we time

         | AR step 2 onward (warm-up dropped)
         | ``total(w/o finalize)`` per step
         | Median of 5 repeat runs
         | p95 reported when spread > 5 %

         .. container:: fd-info-card-title

            What we do **not** report

         | Mean (allocator outliers skew it)
         | Time-to-first-frame
         | Quality metrics (FVD, CLIP-T)

         .. container:: fd-info-card-title

            Reproducibility

         | Every row → one CLI template above
         | Stdout is the source of truth
         | Hand-aggregated until the
         | harness lands

      .. admonition:: PLACEHOLDER — measurement harness
         :class: placeholder

         **What goes here:** wrapper script(s) under
         ``scripts/benchmarks/`` that parse logs into the CSVs that
         back the Results tables. Until that lands, every row is
         hand-computed from the recipe's stdout.

Hardware
--------

.. container:: fd-eyebrow

   Three test systems

.. container:: fd-lede

   FlashDreams is profiled across A100, H100 SXM5, and GB200 NVL72
   nodes. The full hardware specification per row is captured below;
   when the unredacted CPU / RAM / NIC values land, the
   placeholders are replaced in-place.

.. container:: fd-compare

   .. list-table::
      :header-rows: 1
      :widths: 14 14 12 16 16 14 14

      * - System
        - GPU
        - Count
        - NVLink topology
        - Host CPU
        - Host RAM
        - Network
      * - A100 node
        - A100 80GB SXM4
        - 8
        - NVLink 3 fully connected
        - TBD
        - TBD
        - TBD
      * - H100 node
        - H100 80GB SXM5
        - 8
        - NVLink 4 fully connected
        - TBD
        - TBD
        - TBD
      * - GB200 NVL72
        - GB200
        - TBD
        - NVLink 5 (NVL72 fabric)
        - Grace
        - TBD
        - TBD

.. admonition:: PLACEHOLDER — exact hardware spec
   :class: placeholder

   **What goes here:** unredacted CPU model, RAM size, NIC, and slot
   count for each system row above.

   **Source data:** ``lscpu``, ``nvidia-smi topo -m``, ``ibstat`` on
   the test rigs.

   **Reproduce with:** scripts under ``scripts/sysinfo/`` (TBD).

Software
--------

.. container:: fd-eyebrow

   Versions pinned per campaign

.. container:: fd-lede

   Numbers labelled ``main`` track the rolling tip; numbered rows
   below the table will track tagged releases as those land.

.. container:: fd-compare

   .. list-table::
      :header-rows: 1
      :widths: 30 30 40

      * - Component
        - Version
        - Notes
      * - PyTorch
        - TBD
        - CUDA build matching the driver below.
      * - CUDA toolkit
        - TBD
        - Compiled-against version, not driver-reported.
      * - NVIDIA driver
        - TBD
        - ``nvidia-smi`` reading.
      * - Transformer Engine
        - TBD
        - Used by recipes that opt into FP8.
      * - Container image
        - TBD
        - See project README for the public equivalent.
      * - FlashDreams
        - ``main``
        - Numbers labelled "main" track the rolling tip.

.. admonition:: PLACEHOLDER — software versions
   :class: placeholder

   **What goes here:** exact pinned versions of every row above for
   each measurement campaign.

   **Source data:** ``pip freeze`` from the active env; ``nvidia-smi``
   for the driver line.

   **Reproduce with:** the container image referenced in the row of
   the same name.

Results — autoregressive
------------------------

.. container:: fd-eyebrow

   Steady-state step (ms)

.. container:: fd-lede

   Median and 95th-percentile step latency across the AR-2-onward
   window, per recipe. Lower is better.

.. container:: fd-compare fd-compare-numeric

   .. list-table::
      :header-rows: 1
      :widths: 36 16 16 16 16

      * - Recipe (runner slug)
        - GPU
        - p50 step (ms)
        - p95 step (ms)
        - Peak mem (GiB)
      * - ``self-forcing-wan2.1-t2v-1.3b-flash``
        - A100 / H100 / GB200
        - —
        - —
        - —
      * - ``causal-forcing-wan2.1-t2v-1.3b-framewise``
        - H100 80GB
        - —
        - —
        - —
      * - ``causal-forcing-wan2.1-t2v-1.3b-chunkwise``
        - H100 80GB
        - —
        - —
        - —
      * - ``fastvideo-causal-wan2.2-t2v-14b``
        - H100 80GB
        - —
        - —
        - —
      * - ``lingbot-world-fast-flash``
        - H100 80GB
        - —
        - —
        - —
      * - ``omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae``
        - H100 80GB
        - —
        - —
        - —

.. admonition:: PLACEHOLDER — autoregressive step latencies
   :class: placeholder

   **What goes here:** populate the ``—`` cells with median /
   95th-percentile of AR-2-onward ``total(w/o finalize)`` over five
   runs each.

   **Source data:** stdout of each runner; seed numbers for
   ``self-forcing`` exist in ``PERFORMANCE.md`` (single-run, not yet
   aggregated).

   **Reproduce with:** *Methodology* CLI #1 for each runner slug.

Results — bidirectional
-----------------------

.. container:: fd-eyebrow

   End-to-end (s)

.. container:: fd-lede

   Wall-clock for one full denoise pass at each
   ``(GPU, count, resolution)`` cell. Bidirectional recipes are used
   as parity references for the streaming variants above.

.. container:: fd-compare fd-compare-numeric

   .. list-table::
      :header-rows: 1
      :widths: 32 14 14 14 12 14

      * - Recipe (runner slug)
        - GPU
        - GPU count
        - Resolution
        - Frames
        - Wall-clock (s)
      * - ``wan21-t2v-1.3b-480p``
        - H100 80GB
        - 1 / 8
        - 480p
        - —
        - —
      * - ``wan21-i2v-14b-480p``
        - H100 80GB
        - 8
        - 480p
        - —
        - —
      * - ``cosmos2-i2v-2b-720p``
        - H100 80GB
        - 1
        - 720p
        - —
        - —
      * - ``cosmos2-t2v-2b-720p``
        - H100 80GB
        - 1
        - 720p
        - —
        - —

.. admonition:: PLACEHOLDER — bidirectional end-to-end numbers
   :class: placeholder

   **What goes here:** wall-clock seconds for one full denoise pass at
   each ``(GPU, count, resolution)`` cell.

   **Source data:** runner stdout timer line for each bidirectional
   recipe in the Results table.

   **Reproduce with:** *Methodology* CLI #1 (single GPU) or CLI #3
   (multi-GPU / ring attention).

Charts
------

.. container:: fd-eyebrow

   Sweeps, not single points

.. container:: fd-lede

   Three charts ship alongside the tables once the sweep harness
   lands. Until then, each placeholder below names the CSV the chart
   will be drawn from and the methodology row it backs.

.. container:: fd-media-rail

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — throughput vs batch size
         :class: placeholder

         **Chart:** line, x = batch size {1, 2, 4, 8}, y = frames / s.
         One series per recipe in *Results — autoregressive*.
         H100 80GB only.

         **Source CSV:** ``scripts/benchmarks/throughput_sweep.py``
         (TBD). **Reproduce with:** *Methodology* CLI #2 swept across
         ``--batch-size {1,2,4,8}``.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — scaling vs GPU count
         :class: placeholder

         **Chart:** line with two y-axes — throughput (frames / s,
         left) and efficiency (% of linear, right). x = GPU count
         {1, 2, 4, 8}. Bidirectional only.

         **Source CSV:** ``scripts/benchmarks/scaling_sweep.py``
         (TBD). **Reproduce with:** *Methodology* CLI #3 swept across
         ``--nproc_per_node {1,2,4,8}``.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — latency vs resolution
         :class: placeholder

         **Chart:** line, x = resolution {480p, 720p, 1080p}, y = p50
         step latency (ms). One series per autoregressive recipe;
         H100 80GB only.

         **Source CSV:** ``scripts/benchmarks/resolution_sweep.py``
         (TBD). **Reproduce with:** *Methodology* CLI #1 with the
         resolution flag varied per-recipe.

Versus upstream
---------------

.. container:: fd-eyebrow

   Same recipe, same GPU, different runner

.. container:: fd-lede

   The FlashDreams runner calls the same model code paths as the
   upstream library it integrates with, but in a different inference
   environment: KV caches managed by ``flashdreams.infra``, ring
   attention provided by ``flashdreams.core``, CUDA graph captured
   per recipe. Lower is better.

.. container:: fd-compare fd-compare-numeric

   .. list-table::
      :header-rows: 1
      :widths: 30 16 18 18 18

      * - Recipe (runner slug)
        - GPU
        - FlashDreams p50 (ms)
        - Upstream p50 (ms)
        - Ratio
      * - ``self-forcing-wan2.1-t2v-1.3b-flash``
        - A100 80GB
        - —
        - —
        - —
      * - ``self-forcing-wan2.1-t2v-1.3b-flash``
        - H100 80GB
        - —
        - —
        - —
      * - ``causal-forcing-wan2.1-t2v-1.3b-framewise``
        - H100 80GB
        - —
        - —
        - —

.. admonition:: PLACEHOLDER — upstream comparison values
   :class: placeholder

   **What goes here:** fill ``FlashDreams p50`` from the *Results —
   autoregressive* table above; collect ``Upstream p50`` by running
   *Methodology* CLI #4 in the upstream env; ``Ratio`` = upstream /
   FlashDreams.

   **Source data:** seed numbers for ``self-forcing`` exist in
   `PERFORMANCE.md
   <https://github.com/NVIDIA/flashdreams/blob/main/PERFORMANCE.md>`_;
   other rows are TBD.

   **Reproduce with:** *Methodology* CLI #4 paired with #1 on the same
   GPU + driver.

.. admonition:: A note on parity
   :class: fd-callout

   Three integrations currently ship a parity test against their
   upstream reference: ``self_forcing``, ``cosmos_predict2``, and
   ``lingbot`` (each has a ``tests/parity_check/run.sh`` intended for
   manual execution in the upstream env, not in CI). Other recipes
   ship smoke tests only. The numbers on this page assume parity
   holds where it's enforced; see :doc:`../community/index` for how
   to escalate a regression.

How we got here
---------------

.. container:: fd-eyebrow

   Pointers into the rest of the project

.. container:: fd-lede

   This page is the numbers. The architectural decisions behind them
   — and the channels for reporting a regression — live elsewhere on
   the site.

- The :doc:`/api/index` orients you to the four library surfaces and
  links to the design notes for ring attention, KV-cache management,
  and CUDA-graph capture of the steady-state forward.
- The :doc:`/developer_guides/index` cover the architectural concerns
  behind the recipes you can run today; *interactive serving* and
  *new recipes* are the shortest paths to a reproducible local
  measurement of your own.
- `PERFORMANCE.md <https://github.com/NVIDIA/flashdreams/blob/main/PERFORMANCE.md>`_
  is the rolling perf narrative: it carries the raw stdout from the
  three GPUs we've profiled so far, before the numbers were
  aggregated into this page.
- :doc:`../community/index` lists the channels to use if a number on
  this page does not reproduce on your hardware — please file an
  issue rather than averaging away a discrepancy.
