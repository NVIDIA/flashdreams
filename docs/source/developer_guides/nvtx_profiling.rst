.. SPDX-FileCopyrightText: Copyright (c) 2026 Praneeth Samineni.
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

Profiling with Nsight Systems
=============================

Nsight Systems puts the model thread, the UI thread and the GPU on a single
timeline, so you can see when each stage runs, which stages overlap, and where
time is spent waiting. If you only need per-stage timings, ``--stats-path`` is
simpler.

Collecting a report
-------------------

Install the model package and launch the app through the profiler:

.. code-block:: bash

   uv sync --package flashdreams-self-forcing
   uv run flashdreams-profile -- t2v-self-forcing-wan2.1-t2v-1.3b --timeout 300 \
       --output-path artifacts/run.mp4 -- --prompt "A city street at night" --total-blocks 7

By default the report is saved under ``artifacts/profiles/`` and the trace
captures ``cuda,nvtx,osrt,vulkan``. Use ``--report-path PATH`` and ``--trace``
to change either. The example passes the runner's ``--timeout`` because the t2v
app keeps its window open after the last block, and the timeout ends the run
instead.

Leave ``--stats-path`` and ``FLASHDREAMS_SYNC_AND_PROFILE=1`` off while
tracing. Each of them adds two ``torch.cuda.synchronize()`` calls per step, and
those calls change the timeline you are trying to measure.

Reading the report
------------------

Open the ``.nsys-rep`` file in the Nsight Systems GUI to explore the timeline,
or summarize the annotated stages from the terminal:

.. code-block:: bash

   nsys stats --report nvtx_sum artifacts/profiles/<report>.nsys-rep

Each row is one NVTX range, showing how many times it ran and how long it took
on the CPU timeline. Ranges nest, so a parent's duration overlaps its children.
These durations also include time spent waiting, so they do not measure GPU
execution. To see the GPU work a range launched, run the same command with
``--report nvtx_gpu_proj_sum`` and match the rows by range name.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Range
     - Meaning
   * - ``model.pace``
     - Waiting to hold the target step rate.
   * - ``model.step[i]``
     - One autoregressive step.
   * - ``model.publish``
     - Handing the chunk to the presentation queue.
   * - ``pipeline.encode`` / ``diffuse`` / ``decode``
     - Conditioning encode, denoising loop, VAE decode. The names match the
       stage timings ``finalize`` returns.
   * - ``pipeline.finalize``
     - Deferred KV-cache update for the next step.
   * - ``ui.step``
     - One UI loop iteration on the main thread.

The ranges are added in the shared runtime and pipeline, so individual
schedulers do not add any of their own. FlashVSR and SwiftVR provide their own
``generate`` implementations, so their traces do not include the
``pipeline.*`` ranges.
