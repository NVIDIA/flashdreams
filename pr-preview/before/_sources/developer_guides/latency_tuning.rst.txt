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

Interactive-drive latency tuning
================================

``interactive-drive`` latency has two different components:

- **Model / chunk latency** is the time spent preparing HDMap conditioning,
  running the OmniDreams DiT, decoding the generated chunk, and updating model
  state. This is usually the dominant input-to-visual delay.
- **Video-transport latency** is the time spent delivering already-generated
  frames to a local window or browser. Switching from MJPEG to WebRTC can reduce
  this delivery cost, but it does not make the model generate a chunk faster.

Tune the model path first when the profiler shows per-chunk work dominating.
Tune transport when generated frames are ready quickly but arrive late or unevenly
in the viewer.

Model and backend choice
------------------------

Use the OmniDreams world-model backend for latency work. The raster backend is
useful for scene, control, and presenter debugging, but it does not exercise the
model path and should not be used as a model-latency reference.

The packaged interactive-drive manifests are the supported starting points:

- ``example_world_model.yaml`` is the default single-view configuration at
  ``1280 x 704`` (width x height), 30 FPS, 8 generated frames per steady-state
  block, LightVAE enabled, and native DiT acceleration disabled.
- ``example_world_model_perf.yaml`` is the perf-tuned manifest. It lowers the
  default resolution to ``1168 x 640``, keeps 30 FPS and 8-frame steady-state
  blocks, enables the performance recipe, and requires the native DiT path.

Run the perf manifest only on hosts that can build and load the native extension:

.. code-block:: bash

   uv run --package flashdreams-omnidreams omnidreams-prepare --perf
   uv run --package flashdreams-omnidreams interactive-drive \
       --manifest example_world_model_perf.yaml

``native_dit_acceleration: required`` is intentional. If the native extension is
not available, startup fails instead of silently falling back to the slower
PyTorch path.

Resolution
----------

Resolution is one of the highest-impact latency knobs because it changes the
amount of HDMap, DiT, and VAE work per chunk. Set it in the world-model manifest:

.. code-block:: yaml

   resolution_wh: [1168, 640]

Both values are ``[width, height]`` and must be positive multiples of 16. The
bundled manifests list tested aspect-compatible examples:

- ``[1280, 704]``
- ``[1168, 640]``
- ``[1024, 560]``
- ``[896, 496]``
- ``[640, 352]``

Lowering resolution reduces per-chunk compute and transport payload size, with
the expected image-quality tradeoff. The raster presenter automatically adopts
the manifest resolution for the world-model backend.

Chunk size constraints
----------------------

Do not treat chunk size as an arbitrary latency knob. The interactive-drive
adapter validates the FlashDreams pipeline at startup:

- The initial conditioning chunk is fixed at 5 frames.
- The public LightVAE single-view recipe used by the bundled manifests supports
  8-frame steady-state chunks.
- Full-VAE single-view recipes support 8- or 12-frame chunks only when the
  matching checkpoint is available.
- The pixel-shuffle single-view branch requires 16-frame chunks and
  ``local_attn_size: 8``. It is not the published interactive-drive tuning path.

At 30 FPS, an 8-frame steady-state chunk covers about 267 ms of generated video.
Reducing video-transport latency cannot remove this model-side chunk granularity.

FP8 and native acceleration
---------------------------

The perf manifest uses the OmniDreams single-view native CUDA extension for the
DiT path:

.. code-block:: yaml

   native_dit_acceleration: required
   native_dit_backend: fp8_kvcache_cudnn
   native_dit_attention_backend: cudnn

Supported manifest values are ``native_dit_acceleration: disabled | auto |
required`` and ``native_dit_backend: fp8_kvcache_cudnn | bf16``. The attention
backend accepts ``auto``, ``cudnn``, ``sparge``, ``sage3``, and ``sage3_fp8``;
the bundled perf manifest pins ``cudnn``.

The native extension requires a source checkout, ``git``, a CUDA toolchain
(``nvcc``) matching the PyTorch build, synced third-party sources from
``omnidreams-prepare --perf``, and a Blackwell-class GPU (SM 12.0) or newer.
The extension builds for ``12.0a`` by default. Use this path on Blackwell and
GB300 systems.

H100 / Hopper systems should use the standard PyTorch CUDA path with native DiT
disabled unless you are deliberately maintaining a compatible native build. That
path is supported, but it is not the same perf path as the published GB300
numbers.

The manifest also exposes an optional native LightVAE FP8 encoder:

.. code-block:: yaml

   native_vae_encoder: fp8

It is disabled in the bundled perf manifest. Enabling it requires
``OMNIDREAMS_LIGHTVAE_FP8_STATE_PATH`` or ``native_vae_fp8_state_path`` pointing
to a calibrated LightVAE FP8 state.

Transport choice
----------------

Pick transport based on where the viewer runs:

- Local Vulkan window: lowest-overhead local presentation when the host has a
  graphics-capable GPU and display stack.
- ``--stream-mjpeg [HOST:]PORT``: simple browser delivery from the same process.
  Use it on compute-only hosts such as GB300 systems without a graphics queue,
  or when a laptop browser views a remote model host.
- ``omnidreams.webrtc.server``: richer browser frontend with WebRTC's lower
  video-delivery latency and streaming gRPC service support. Prefer this for
  product-style remote viewing or multi-client integration.

MJPEG and WebRTC affect video delivery after a frame exists. If the model is
still spending most of the time inside each chunk, use the perf manifest,
resolution, and native-acceleration knobs first.

Profiling and validated reference
---------------------------------

Use ``--profile-world-model`` to enable FlashDreams CUDA-event profiling for the
world-model runtime. Use ``--sync-gpu-timing`` only when you need raster compute
timings; it synchronizes GPU work and is not a throughput setting.

The validated published reference for interactive-drive latency is the
single-view GB300 table from :doc:`/models/omnidreams`, measured at
``1280 x 704`` resolution:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18 18

   * - Stage
     - 1x GPU
     - 2x GPU
     - 4x GPU
     - 8x GPU
   * - HDMap Encoder
     - 28 ms
     - 26 ms
     - 26 ms
     - 26 ms
   * - Diffusion DiT
     - 84 ms
     - 71 ms
     - 49 ms
     - 47 ms
   * - VAE Decoder
     - 6 ms
     - 5 ms
     - 5 ms
     - 5 ms
   * - KV-cache Update
     - 42 ms
     - 34 ms
     - 23 ms
     - 22 ms
   * - **Total**
     - **118 ms**
     - **102 ms**
     - **80 ms**
     - **78 ms**
   * - **Effective FPS**
     - **68**
     - **78**
     - **100**
     - **103**

KV-cache update is off the hot path and excluded from the total. This guide
consolidates the supported latency controls and existing published measurement;
it does not add new end-to-end hardware benchmarking for Hopper, H100,
Blackwell, or GB300 systems.
