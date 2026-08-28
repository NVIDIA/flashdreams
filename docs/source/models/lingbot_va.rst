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

LingBot-VA
==========

.. container:: fd-cta-row

   .. button-link:: https://technology.robbyant.com/lingbot-va
      :color: primary

      Project page

   .. button-link:: https://arxiv.org/abs/2601.21998
      :color: primary

      arXiv paper

   .. button-link:: https://github.com/Robbyant/lingbot-va
      :color: primary

      Official code

   .. button-link:: https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin
      :color: primary

      Checkpoint

LingBot-VA is an autoregressive diffusion video-action world-model policy. It
uses a shared video/action backbone to predict visual dynamics and robot actions.
The FlashDreams integration implements the pinned RoboTwin image-to-video-action
(I2AV) path as an offline, batch-one rollout. It does not implement the upstream
closed-loop observation-feedback or asynchronous motor-execution system.

This limitation is specific to the FlashDreams adapter. The pinned upstream
repository also contains a
`RoboTwin evaluator
<https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/evaluation/robotwin/eval_polict_client_openpi.py#L542-L609>`_
and `model server
<https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/wan_va_server.py#L572-L627>`_
that execute action chunks, capture actual simulator observations, and feed the
observations and executed state back into the model cache. That environment and
execution bridge is not included here.

Supported FlashDreams method
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Integrated behavior
   * - Application slug
     - ``lingbot-va-robotwin-i2av``
   * - Task
     - One instruction and three RGB camera observations to a predicted
       high-camera video and denormalized RoboTwin actions
   * - Inputs
     - Natural-language prompt plus high, left-wrist, and right-wrist PNGs
   * - Outputs
     - TCHW video at 10 FPS, float32 ``actions[step, channel]``, and numeric
       inference metrics
   * - Execution
     - BF16 on one CUDA device, with optional ``torch.compile`` and component
       offload; one complete rollout per engine
   * - Checkpoint
     - ``robbyant/lingbot-va-posttrain-robotwin`` pinned to
       ``8c9dea8abbc5c91cc9e18bc3264b8915083bbe70``

Requirements
------------

- **Python**: 3.10 or newer.
- **Runtime dependencies**: Diffusers 0.38.x and Transformers 5.x. The
  integration uses private Wan VAE streaming fields, so widening the Diffusers
  range requires a real-model retest.
- **GPU validation**: one NVIDIA RTX PRO 6000 Blackwell Workstation Edition
  (97,887 MiB), BF16. The measured two-chunk peak was 37.07 GiB with component
  offload and 40.35 GiB with components resident. These are measurements, not
  guaranteed minimum-VRAM requirements.
- **Checkpoint storage**: approximately 22.7 GiB of resolved weight files.
- **Input data**: three camera images are required and are not bundled.

Installation
------------

From the repository root:

.. code-block:: bash

   uv sync --project integrations_v2/lingbot_va

Running the method
------------------

.. code-block:: bash

   uv run --project integrations_v2/lingbot_va flashdreams-run-v2 \
       lingbot-va-robotwin-i2av \
       --mode mp4 \
       --output-path outputs/lingbot_va/demo.mp4 \
       --stats-path outputs/lingbot_va/metrics.json \
       --tensor-artifact-dir outputs/lingbot_va \
       -- \
       --checkpoint-root robbyant/lingbot-va-posttrain-robotwin \
       --checkpoint-revision 8c9dea8abbc5c91cc9e18bc3264b8915083bbe70 \
       --input-image-dir /path/to/robotwin-images \
       --num-chunks 10

Use ``--no-compile`` for correctness debugging and ``--enable-offload`` when
GPU memory is constrained. Run the following to list every model-specific
option:

.. code-block:: bash

   uv run --project integrations_v2/lingbot_va flashdreams-run-v2 \
       lingbot-va-robotwin-i2av -- --help

Inputs and outputs
------------------

The input directory must contain these exact filenames:

- ``observation.images.cam_high.png``
- ``observation.images.cam_left_wrist.png``
- ``observation.images.cam_right_wrist.png``

The high camera is encoded at 256x320. Each wrist camera is encoded at 128x160;
their latents form the upper bar of the upstream RoboTwin T layout.

Let ``N`` be the positive ``--num-chunks`` value. One model step returns:

- floating video ``[8N - 3, 3, 256, 320]`` in ``[-1, 1]`` at 10 FPS;
- float32 actions ``[32N, 16]``, selected in channel order
  ``0..6, 28, 7..13, 29``;
- prompt, observation, denoise, decode, total, and peak-allocation metrics.

MP4, NumPy action, and JSON metric serialization are provided by generic V2
runtime sinks rather than model-specific file handling.

The 16 action columns are relative two-arm RoboTwin commands:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Columns
     - Meaning
   * - ``0..2``
     - Left end-effector x/y/z translation delta
   * - ``3..6``
     - Left relative quaternion x/y/z/w
   * - ``7``
     - Left gripper command
   * - ``8..10``
     - Right end-effector x/y/z translation delta
   * - ``11..14``
     - Right relative quaternion x/y/z/w
   * - ``15``
     - Right gripper command

The upstream evaluator composes the relative poses with the episode's initial
end-effector poses and normalizes the resulting quaternions before simulator
execution. FlashDreams emits the denormalized relative values and does not
perform pose composition or actuation.

Inspecting action outputs
-------------------------

The integration supplies a model-specific inspector that validates a committed
tensor-artifact directory, plots both arms' translation/quaternion/gripper
channels, and can export named CSV columns:

.. code-block:: bash

   uv run --project integrations/lingbot_va --extra visualization \
       lingbot-va-visualize-actions outputs/lingbot_va \
       --output outputs/lingbot_va/actions.png \
       --csv-output outputs/lingbot_va/actions.csv

A direct ``actions.npy`` path from an older validation run is also accepted.
The plot is a diagnostic for inspection and batch comparison; it is not a task
success, physical-validity, or robot-safety evaluation.

Model details
-------------

.. list-table::
   :header-rows: 1
   :widths: 27 73

   * - Component
     - Integrated configuration
   * - Text encoder
     - UMT5-XXL; 4,096-wide states; at most 512 tokens
   * - VAE
     - Wan VAE; 48 latent channels; 16x spatial and 4x temporal scaling
   * - Video-action DiT
     - 5,088,872,670 parameters; 30 shared blocks; width 3,072; FFN width
       14,336; 24 attention heads of width 128
   * - Video patches
     - ``[1, 2, 2]``
   * - Chunk geometry
     - Two latent frames; 240 video tokens and 32 action tokens per chunk
   * - Attention cache
     - 36 autoregressive slots per conditional or unconditional branch;
       9,792 tokens per block and branch
   * - Guidance defaults
     - Video CFG 5; action CFG 1
   * - Checkpoint footprint
     - 9.48 GiB transformer, 10.58 GiB text encoder, and 2.63 GiB VAE,
       measured from the pinned snapshot

The upstream RoboTwin setting ``attn_window=72`` counts alternating video and
action regions: two regions form one autoregressive cache slot. The divisor is
therefore independent of the two latent frames in a chunk.

Intended use and safety
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Intended and validated
     - Outside this integration's validated scope
   * - Research and development of offline RoboTwin video/action rollout
       inference
     - Direct, unattended, or safety-critical robot actuation
   * - Numerical parity, lifecycle, packaging, and V2 runtime regression tests
     - Claims of physical accuracy, task success, or safe action execution
   * - Generating inspectable video, action, and metric artifacts
     - Training, fine-tuning, LIBERO checkpoints, multi-GPU/FSDP, or online
       policy serving

Predicted actions can be wrong, temporally inconsistent, or unsafe. Distribution
shift, camera calibration, prompt ambiguity, and accumulated autoregressive
error can affect both modalities. Evaluate in simulation or a
hardware-interlocked environment with task-specific limits and human oversight
before considering physical use.

Data, evaluation, and provenance
--------------------------------

The architecture and inference behavior are ported from
`Robbyant/lingbot-va at 7c6ffa9
<https://github.com/Robbyant/lingbot-va/tree/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb>`_.
Upstream identifies
`robotwin-clean-and-aug-lerobot
<https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot>`_
as its cleaned and augmented RoboTwin post-training dataset. FlashDreams does
not train, alter, or independently audit the checkpoint or dataset.

The `LingBot-VA paper <https://arxiv.org/abs/2601.21998>`_ reports simulated
and real-robot results. This integration has not reproduced its task-success
rates. The evidence below establishes native-flow parity, output contracts, and
system execution for the pinned checkpoint.

Validation evidence
-------------------

The pinned checkpoint contains 841 transformer entries. Two obsolete
``patch_embedding.*`` entries are dropped; the remaining 839 entries map
bijectively to the native model's 839 entries and load strictly.

Matched first-chunk upstream/native comparisons use maximum absolute error
``<= 0.07`` and mean absolute error ``<= 0.012`` as acceptance gates.

.. list-table::
   :header-rows: 1
   :widths: 18 14 22 22 22

   * - Native mode
     - Stream
     - Maximum absolute error
     - Mean absolute error
     - RMS error
   * - eager
     - video
     - 0.04296875
     - 0.00751040
     - 0.00951632
   * - eager
     - action
     - 0.06250000
     - 0.00875314
     - 0.01267146
   * - compiled
     - video
     - 0.05468750
     - 0.00970979
     - 0.01229867
   * - compiled
     - action
     - 0.06250000
     - 0.01116651
     - 0.01560142

Matched resident and offloaded two-chunk runs used default CFG, 25 video steps,
50 action steps, no compilation, one CUDA device, and seed 42.

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 18 18

   * - Mode
     - Prompt + observation
     - Denoise
     - Total
     - Peak allocation
   * - resident
     - 0.461 s
     - 4.735 s
     - 33.220 s
     - 40.35 GiB
   * - offload
     - 5.831 s
     - 4.744 s
     - 34.009 s
     - 37.07 GiB

Both runs returned finite video ``[13, 3, 256, 320]`` and actions
``[64, 16]`` with byte-identical outputs. Final post-rebase revalidation passed
in 35.82 seconds, with model-reported total 31.516 seconds and peak allocation
39,804,413,440 bytes. The action SHA-256
remained
``463b307b667c1ca13a47bbbc5a17f68604621dfe3c3a10fc5860077216928d95``.

These are implementation measurements from 2026-08-25 and 2026-08-27, not
general model-performance or robot-success claims. Full input hashes,
reproduction commands, phase timings, MP4 metadata, and architectural diagrams
are maintained in the
`LingBot-VA integration README
<https://github.com/NVIDIA/flashdreams/blob/main/integrations/lingbot_va/README.md>`_.

Limitations and license
-----------------------

The integration supports one GPU, one complete deferred-decode rollout, and the
RoboTwin checkpoint above. It makes no FSDP, context-parallel, live-control,
online-serving, or per-chunk presentation claim.

The FlashDreams packages are Apache-2.0. The upstream repository and checkpoint
model card also identify Apache-2.0. Users remain responsible for checkpoint and
dataset terms.

Citation
--------

If you use LingBot-VA, cite the original work:

.. code-block:: bibtex

   @article{lingbot-va2026,
     title={Causal World Modeling for Robot Control},
     author={Li, Lin and Zhang, Qihang and Luo, Yiming and Yang, Shuai and
       Wang, Ruilin and Han, Fei and Yu, Mingrui and Gao, Zelin and Xue, Nan
       and Zhu, Xing and Shen, Yujun and Xu, Yinghao},
     journal={arXiv preprint arXiv:2601.21998},
     year={2026}
   }
