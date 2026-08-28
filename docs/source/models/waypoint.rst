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

Waypoint 1.5
============

.. container:: fd-cta-row

   .. button-link:: https://huggingface.co/Overworld/Waypoint-1.5-1B
      :color: primary

      Checkpoint and upstream model card

   .. button-link:: https://github.com/Overworldai/world_engine
      :color: primary

      Official inference code

   .. button-link:: https://github.com/Overworldai/Biome
      :color: primary

      Official desktop client

Waypoint-1.5-1B is Overworld's dense, autoregressive interactive video world
model. FlashDreams integrates the published BF16 checkpoint as an
image-established, keyboard/mouse-controlled V2 application with deterministic
MP4 replay, per-action metrics, and live WebRTC presentation.

.. raw:: html

   <div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
     <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
       <source src="https://huggingface.co/Overworld/Waypoint-1.5-1B/resolve/main/assets/wp_1.5.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>
   <p class="model-footnote">
     Upstream Waypoint 1.5 teaser from the
     <a href="https://huggingface.co/Overworld/Waypoint-1.5-1B">Overworld model card</a>;
     this is not a FlashDreams benchmark artifact.
   </p>

Support summary
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Surface
     - FlashDreams support
   * - Application slug
     - waypoint-1-5-1b through flashdreams-run-v2
   * - Input modalities
     - RGB/RGBA seed image; keyboard and mouse buttons; relative mouse motion;
       ternary scroll-wheel direction
   * - Output modality
     - Four RGB frames per action, native 1024x512 TCHW in the [-1, 1] range
   * - Control modes
     - Live browser events or a finite, versioned JSON action timeline
   * - Output modes
     - WebRTC, MP4, null output, and optional per-action metrics
   * - Precision and device
     - BF16 on CUDA; no FlashDreams quantized or CPU inference path
   * - Not implemented
     - Text prompting, the 360P checkpoint, quantization, and multi-GPU execution

The checkpoint's pinned configuration sets prompt_conditioning to null.
Although the generic upstream APIs accept prompts for other models, this
checkpoint-compatible FlashDreams path has no text encoder or prompt-conditioned
weights and deliberately exposes no prompt argument. In the upstream runtime,
this configuration leaves the prompt encoder uninitialized and calling
``set_prompt`` raises. The pinned safetensors artifact contains 393 tensor keys,
with no keys for prompt or cross-attention modules. Consequently, this model
does not accept a text prompt that can influence its output.

Requirements
------------

- A CUDA-capable NVIDIA GPU with BF16 and PyTorch FlexAttention support.
- The FlashDreams path was validated on one NVIDIA RTX PRO 6000 Blackwell
  Workstation Edition. The measured PyTorch peak allocation was 6.061 GiB, but
  this is not a minimum-VRAM guarantee and excludes non-PyTorch process memory.
- The first run downloads the 3.72 GB BF16 Waypoint safetensors file and the
  separate Overworld-Models/taehv1_5 checkpoint into the Hugging Face cache.
- ffmpeg on PATH is required for MP4 output.

Installation
------------

From the FlashDreams repository root:

.. code-block:: bash

   uv sync --package flashdreams-waypoint-v2 --inexact

The V2 application package depends on the sibling flashdreams-waypoint model
package, so both are installed together.

Running the model
-----------------

Generate a deterministic 40-action MP4 using the pinned public example image
and bundled control timeline:

.. code-block:: bash

   uv run --no-sync flashdreams-run-v2 waypoint-1-5-1b \
       --output-path waypoint.mp4 --stats-path waypoint.metrics.json \
       -- --example-data --actions 40 --seed 464 --profile

Run the same application interactively in a browser:

.. code-block:: bash

   uv run --no-sync flashdreams-run-v2 waypoint-1-5-1b \
       --mode webrtc --host 127.0.0.1 --port 8766 \
       -- --seed-image seed.png --seed 464

Open http://127.0.0.1:8766/. Arguments before the separator configure the V2
runtime; arguments after it configure Waypoint. To inspect all model arguments:

.. code-block:: bash

   uv run --no-sync flashdreams-run-v2 waypoint-1-5-1b -- --help

Model and integration architecture
----------------------------------

The pinned checkpoint configuration and FlashDreams implementation agree on
these model-facing invariants:

- Dense 24-block autoregressive diffusion transformer, width 2048.
- 32 query heads, 16 K/V heads, and a four-times-width feed-forward network.
- One model action is a 32-channel, 32x64 latent frame patchified into 512
  spatial tokens.
- Four rectified-flow Euler evaluations at sigmas 1.0, 0.9, 0.75, and 0.3;
  the terminal 0.0 evaluation commits clean cache state.
- Control conditioning uses 256 button IDs, two mouse-delta values, and one
  ternary wheel value. Control fusion is present every third block.
- Most blocks retain 16 latent actions densely. Blocks 3, 7, 11, 15, 19, and
  23 use a 128-action horizon with every eighth historical action pinned.
- TAEHV encodes four seed RGB frames into one latent action and decodes every
  generated latent action into four RGB frames.

The 128-action global horizon corresponds to 512 presented RGB frames, matching
the context length stated by the upstream model card. FlashDreams presents the
codec's native 1024x512 canvas. The official world_engine client can instead
resize 1280x720 input to that native canvas and resize decoded output back to
1280x720; FlashDreams intentionally omits those extra spatial resamples.

The upstream model card advertises a **1.2B parameter count**. Independently,
the pinned model.safetensors header contains **1,860,823,096 BF16 tensor
elements across 393 tensors** (3,721,694,304 bytes). These are different
published-model versus serialized-checkpoint accounting figures; FlashDreams
reports both rather than relabeling the upstream model.

The package-level design review contains component, class, use-case, and
sequence diagrams:

.. button-link:: https://github.com/NVIDIA/flashdreams/blob/main/integrations/waypoint/README.md
   :color: secondary
   :outline:

   Waypoint architecture design

Measured FlashDreams performance
--------------------------------

The final FlashDreams path was measured on 2026-08-26 using an RTX PRO 6000
Blackwell Workstation Edition (96 GiB), driver 595.84, PyTorch 2.12.1+cu130,
CUDA 13.0, BF16 weights, the pinned example seed/control timeline, seed 464,
native 1024x512 output, four denoise evaluations, and synchronous profiling.
Actions 1-19 were warmup; actions 20-40 were the steady-state sample.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Measurement
     - Result
   * - Mean action latency
     - 68.791 ms per four generated frames
   * - Median action latency
     - 68.887 ms
   * - p90 action latency
     - 69.737 ms
   * - Throughput derived from mean latency
     - 58.15 generated RGB frames/s
   * - Peak PyTorch CUDA allocation
     - 6.061 GiB
   * - Encoded output
     - 164 frames: 4 seed + 40 actions x 4 frames

The declared 60 FPS is presentation timing, not a throughput guarantee.
Overworld separately reports 56 FPS for its unquantized runtime on an RTX 5090;
that result was not reproduced here and is not directly comparable across
different GPU, runtime, and presentation stacks.

Parity and rollout validation
-----------------------------

FlashDreams loaded the same BF16 checkpoint and matched the pinned official
world_engine implementation through one complete controlled action. The final
comparison includes the four-step Euler solve and both cache commits:

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Comparison
     - Mean absolute error
     - Max absolute error
     - Cosine similarity
   * - Seed cache flow
     - 0.010793
     - 0.218750
     - 0.999586
   * - Clean latent after four steps
     - 0.005682
     - 0.033691
     - 0.999869
   * - Final cache flow
     - 0.010511
     - 0.265625
     - 0.999487

The residual is consistent with BF16 execution through different compiled
kernel boundaries. Review optimizations retained a byte-identical 40-action
MP4. Additional inference completed 15 distinct scenes at 40 actions each and
one 118-action rollout: 718 generated actions and 2,936 decoded frames in total,
with complete finite metrics and exact frame accounting. This demonstrates
execution, cache longevity, and scene coverage; it is not a qualitative
gameplay or physical-accuracy score.

.. button-link:: https://github.com/NVIDIA/flashdreams/blob/main/integrations_v2/waypoint/VALIDATION.md
   :color: secondary
   :outline:

   Complete validation record

Intended use and limitations
----------------------------

Waypoint is suitable for research and prototyping around interactive video
worlds, creative exploration, control-conditioned generation, and low-latency
world-model systems. It is a generative model, not a physically grounded
simulator.

Important limitations:

- Long rollouts can drift, collapse, or become inconsistent.
- Geometry, motion, object identity, and persistence can be unstable.
- Outputs may reflect biases or unsafe patterns learned from training data.
- The FlashDreams integration has no content-safety filter of its own.
- It is not appropriate for safety-critical decisions, surveillance,
  high-stakes automation, or deployments that remove reasonable safeguards.
- The current server accepts one WebRTC browser client per process. Multiple
  sessions may share model weights, but model execution is serialized.
- Live results depend on browser event timing. Use a control file and fixed
  seed when reproducibility matters.

Review the upstream model card and world-model safety discussion before
deployment:

.. container:: fd-cta-row

   .. button-link:: https://huggingface.co/Overworld/Waypoint-1.5-1B
      :color: secondary
      :outline:

      Upstream model card

   .. button-link:: https://over.world/blog/engineering-safety-for-interactive-world-models
      :color: secondary
      :outline:

      Upstream safety discussion

Provenance
----------

- Model: Overworld/Waypoint-1.5-1B, revision
  391f92827075edcf4a8b3c8a2ddae010698f8636, Apache-2.0.
- Model SHA-256:
  b872ad07968bae082a120a29072e61a13565086f042384ad7fdb79a7b0c50994.
- TAEHV: Overworld-Models/taehv1_5, revision
  a0253886b13b9c4c3bd224bd479be03f5988a3df.
- Official parity implementation: Overworldai/world_engine at
  b3f1e725dedac17ccbfaf9ee37f5e068bb44bed4.
- FlashDreams integration and documentation: Apache-2.0.
