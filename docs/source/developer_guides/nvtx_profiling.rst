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

Nsight Systems shows the model thread, the UI thread and the GPU on one
timeline, so you can see what each of them was doing at the same moment. If you
only need numbers for each step, such as stage timings, frame rates and input
latency, use ``--stats-path`` instead.

Collecting a report
-------------------

First install the model package, then launch the app through
``flashdreams-profile``:

.. code-block:: bash

   uv sync --package flashdreams-self-forcing
   uv run flashdreams-profile -- t2v-self-forcing-wan2.1-t2v-1.3b --timeout 300 \
       --output-path artifacts/run.mp4 -- --prompt "A city street at night" --total-blocks 7

The report is saved in ``artifacts/profiles/``. To save it somewhere else, pass
``--report-path``. By default the report records CUDA, NVTX, OS runtime and
Vulkan activity; to record something different, pass ``--trace``.

To time each stage, ``--stats-path`` (or ``FLASHDREAMS_SYNC_AND_PROFILE=1``)
makes the CPU wait for the GPU twice per step. This adds a slight delay to each
step, so the timeline differs slightly from a normal run.

Reading the report
------------------

To read the report, open the ``.nsys-rep`` file in the Nsight Systems GUI. For a
quick summary in the terminal instead, run:

.. code-block:: bash

   nsys stats --report nvtx_sum artifacts/profiles/<report>.nsys-rep

Each row of the summary is one of the ranges below, with how many times it ran
and how long it took. These durations are CPU time and include waiting. For the
GPU work that each range launched, use ``--report nvtx_gpu_proj_sum`` instead.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Range
     - Meaning
   * - ``model.pace``
     - Waiting to hold the step rate.
   * - ``model.step[i]``
     - One autoregressive step.
   * - ``model.publish``
     - Handing the chunk to presentation.
   * - ``pipeline.encode`` / ``diffuse`` / ``decode``
     - Encode, denoising loop, VAE decode.
   * - ``pipeline.finalize``
     - KV-cache update for the next step.
   * - ``ui.step``
     - One UI loop iteration.
   * - ``input.wait``
     - From an input arriving to the first frame that shows it.

Besides these ranges, the timeline has a mark each time a frame is generated
(``model.frame``) or shown (``present.frame``). Each mark includes the step that
produced the frame, such as ``present.frame [step 5]``, so you can follow a frame
on screen back to the step that made it.

FlashVSR and SwiftVR use their own ``generate`` method, so their reports have no
``pipeline.*`` ranges.
