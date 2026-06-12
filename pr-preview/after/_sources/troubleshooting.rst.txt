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

Troubleshooting
===================================

Use this page for common first-run failures before opening an issue. Each
entry lists the visible symptom, the most likely cause, and the next concrete
step to try.

CUDA or PyTorch build mismatch
------------------------------

**Symptoms:**

- A CUDA extension fails to build or load.
- ``interactive-drive --manifest example_world_model_perf.yaml`` exits instead
  of falling back to the default PyTorch path.
- Errors mention ``nvcc``, a CUDA version, a GPU architecture, or missing CUDA
  libraries.

**Likely cause:**

The OmniDreams perf manifest uses native DiT and LightVAE acceleration with
``native_dit_acceleration: required``. That path requires a source checkout,
``git``, a CUDA toolchain with ``nvcc`` matching the installed PyTorch build,
and a Blackwell-class GPU (SM 12.0) or newer.

**Fix or next step:**

Start with the non-perf OmniDreams launch in :doc:`/models/omnidreams`. If you
need the perf manifest, prepare the pinned third-party sources first:

.. code-block:: bash

   uv run --package flashdreams-omnidreams omnidreams-prepare --perf

Then verify that the machine has the required GPU and a CUDA toolchain that
matches the PyTorch build before launching:

.. code-block:: bash

   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   nvcc --version

Disk or cache exhaustion
------------------------

**Symptoms:**

- A first run fails while downloading or loading checkpoints.
- The process reports no space left on device, stops mid-load, or leaves a
  partial Hugging Face cache.
- Output video or stats files are missing after an interrupted run.

**Likely cause:**

Model checkpoints and example assets are cached on first use. LingBot-World
downloads a checkpoint of about 70 GB under ``$HF_HOME`` and its docs recommend
keeping about 200 GB free for the model plus Hugging Face cache. Example data
and generated outputs also consume local disk under paths such as
``assets/example_data/lingbot_world/<NN>/`` and ``outputs/``.

**Fix or next step:**

Check free space on both the repository filesystem and the Hugging Face cache
filesystem. If the default cache location is too small, point ``HF_HOME`` at a
larger volume before running the model:

.. code-block:: bash

   export HF_HOME=/path/to/large/cache

Then rerun the same command so the downloader can reuse or repair the cache.

Model download or authentication failure
----------------------------------------

**Symptoms:**

- A download returns 401, 403, not found, or gated-repository errors.
- OmniDreams scene or checkpoint downloads fail before the demo starts.
- LingBot-World fails while fetching the model checkpoint from Hugging Face.

**Likely cause:**

Most model runs need Hugging Face authentication. OmniDreams requires an
``HF_TOKEN`` with read access to the ``nvidia/omni-dreams-scenes`` dataset and
the ``nvidia/omni-dreams-models`` model repository. Other model pages also
document first-run downloads from Hugging Face.

**Fix or next step:**

Export a valid token in the same shell that launches FlashDreams:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

If the token is already set, confirm that the account behind it can open the
model or dataset page referenced by the failing command, then rerun the
FlashDreams command.

GPU out of memory
-----------------

**Symptoms:**

- The run exits with ``CUDA out of memory`` or the Python process is killed
  during model load or generation.
- LingBot-World runs out of memory on a single GPU when using large
  ``--total-blocks`` values.
- Multi-GPU commands fail after one or more ranks report memory pressure.

**Likely cause:**

The selected model, resolution, rollout length, or GPU count does not fit the
available VRAM. The model pages list minimum VRAM expectations: OmniDreams is
about 48 GB, Self-Forcing is about 24 GB, and LingBot-World is about 120 GB.

**Fix or next step:**

Use a smaller documented run first: reduce ``--total-blocks``, lower
``--pixel-height`` and ``--pixel-width`` where the model page exposes those
flags, or use the documented multi-GPU ``torchrun --nproc_per_node=<N>``
launch. For LingBot-World, also try the documented efficient streaming preset
``lingbot-world-fast-taehv-window15-sink3``.

WebRTC connection or video does not appear
------------------------------------------

**Symptoms:**

- ``/request_session`` is not reachable from the browser.
- The page loads but video never appears.
- The server seems idle before printing the connection URL.

**Likely cause:**

The WebRTC servers open their HTTP port only after model load and warmup. On a
remote or cloud GPU instance, the server port may not be reachable directly at
the host IP. If ``/request_session`` loads but video does not appear, the
browser may be hiding local IPs in WebRTC ICE candidates with mDNS hostnames.

**Fix or next step:**

Wait until the server prints ``Connect via http://<server-ip>:8089/request_session``.
For remote machines, forward the documented port and open the local URL:

.. code-block:: bash

   ssh -L 8089:localhost:8089 <user>@<host>

Then open ``http://localhost:8089/request_session``. If the page loads but the
video still does not appear, follow the browser-specific WebRTC setting in
:doc:`/models/omnidreams` or :doc:`/models/lingbot_world`.

Triton autotuning or warmup looks stuck
---------------------------------------

**Symptoms:**

- The first launch takes several minutes.
- Logs mention Triton autotuning or CUDA-graph warmup.
- Later runs are much faster than the first run.

**Likely cause:**

Cold runs include one-time setup. The quickstart and model pages document that
first launches can include downloads, Triton autotuning, CUDA-graph warmup, and
for OmniDreams native acceleration, first-use extension compilation.

**Fix or next step:**

Let the first launch finish if it is still making progress. Subsequent launches
reuse caches. For quick validation, use the small documented demo values such
as ``--total-blocks 7`` for Self-Forcing or inspect a runner without loading the
model by using ``--no-instantiate`` as described below.

``--no-instantiate`` prints a config but does not run
-----------------------------------------------------

**Symptoms:**

- ``flashdreams-run`` prints ``Resolved config for ...`` and exits without
  downloading checkpoints, warming up, or writing an output video.
- No files appear under ``outputs/``.

**Likely cause:**

``--no-instantiate`` is a diagnostic flag. It resolves and prints the runner
configuration, then returns before creating the runner or calling
``runner.run()``.

**Fix or next step:**

Use ``--no-instantiate`` only when you want to confirm that a runner slug and
CLI overrides parse correctly:

.. code-block:: bash

   uv run flashdreams-run --no-instantiate self-forcing-wan2.1-t2v-1.3b-taehv

Remove the flag for a real generation run, or use ``--help`` on the runner
slug to inspect all supported options.
