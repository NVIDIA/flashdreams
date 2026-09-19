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

An Nsight Systems trace records the model thread, the UI thread and the GPU on
one clock. Use it to analyze CPU <-> GPU overlap.

Collecting a report
-------------------

.. code-block:: bash

   uv sync --package flashdreams-self-forcing
   uv run flashdreams-profile -- t2v-self-forcing-wan2.1-t2v-1.3b --timeout 300 \
       --output-path artifacts/run.mp4 -- --prompt "A city street at night" --total-blocks 7

The report lands in ``artifacts/profiles/``. ``flashdreams-profile`` sets
``FLASHDREAMS_NVTX=1`` for the child process; without it the ranges are no-ops
and the trace has no labels. The t2v app keeps its window open after the last
block, so pass the runner's ``--timeout`` to end the run.

``--report-path PATH`` writes the report to ``PATH`` instead. ``--trace``
defaults to ``cuda,nvtx,osrt,vulkan``; the native-window and ImGui renderers
draw through Vulkan, so their submissions land beside the CUDA work.

Reading the report
------------------

Open the ``.nsys-rep`` in the Nsight Systems GUI, or summarize it from the
terminal:

.. code-block:: bash

   nsys stats --report nvtx_sum artifacts/profiles/<report>.nsys-rep
   nsys stats --report nvtx_gpu_proj_sum artifacts/profiles/<report>.nsys-rep

``nvtx_sum`` gives each range's CPU time. ``nvtx_gpu_proj_sum`` projects each
range onto the GPU work it launched, so the two can be compared row by row.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Range
     - Meaning
   * - ``model.pace``
     - Wait for the next cadence slot.
   * - ``model.step[i]``
     - One autoregressive step.
   * - ``model.publish``
     - Publish of the chunk to the presentation queue.
   * - ``pipeline.encode`` / ``diffuse`` / ``decode``
     - Conditioning encode, denoising loop, VAE decode. The names match the
       stage timings :class:`~flashdreams.infra.profiler.EventProfiler` logs.
   * - ``pipeline.finalize``
     - Deferred KV-cache update for the next step.
   * - ``denoise[i]``
     - One denoising iteration: a single transformer forward pass.
   * - ``ui.step``
     - One UI loop iteration on the main thread.

FlashVSR and SwiftVR implement ``generate`` themselves, so they emit no
``pipeline.*`` ranges. Waypoint's scheduler has its own sampling loop and emits
no ``denoise[i]``.

Notes
-----

- Under CUDA graphs, ``denoise[i]`` measures only the launch of the replay.
- Percentages in ``nvtx_sum`` double-count nested ranges: a ``denoise[i]`` is
  also inside ``pipeline.diffuse``, which is inside ``model.step``. Read the
  instance counts and medians instead.
- The ranges call only NVTX. ``--stats-path`` and
  ``FLASHDREAMS_SYNC_AND_PROFILE=1`` synchronize once per stage, so leave them
  off when you are studying CPU <-> GPU overlap.
