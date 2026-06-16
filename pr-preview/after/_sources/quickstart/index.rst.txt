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

Get Started
===========

Welcome to FlashDreams! This page will guide you from a fresh checkout
of the repository to a running model. It uses :doc:`NVIDIA OmniDreams
</models/omnidreams>`, the interactive driving world model, as the
example; the :doc:`model gallery </models/index>` lists the run command
for every other model.

Install
-------

FlashDreams uses the ``uv`` Python package manager (`installation
instructions <https://docs.astral.sh/uv/getting-started/installation/>`_).
With ``uv`` installed, clone the repository and synchronize the
OmniDreams workspace:

.. code-block:: bash

   git clone https://github.com/NVIDIA/flashdreams.git
   cd flashdreams
   uv sync --package flashdreams-omnidreams --extra interactive-drive

Most runs need a Hugging Face token. For OmniDreams, use a token with
read access to `nvidia/omni-dreams-models
<https://huggingface.co/nvidia/omni-dreams-models>`_ and
`nvidia/omni-dreams-scenes
<https://huggingface.co/datasets/nvidia/omni-dreams-scenes>`_:

.. code-block:: bash

   export HF_TOKEN=<your-hf-token>

For container, caching, and other environment details, see the project
`README <https://github.com/NVIDIA/flashdreams/blob/main/README.md>`_ and
:doc:`/troubleshooting`.

Run your first model
--------------------

Launch the OmniDreams interactive driving demo. It runs the world model
and streams the generated camera view to a browser:

.. code-block:: bash

   uv run --package flashdreams-omnidreams interactive-drive --stream-mjpeg :8080

Then open ``http://<server-ip>:8080/`` in a browser on the same network
(use ``localhost`` on the same machine). The first launch spends several
minutes loading checkpoints and compiling kernels; later launches reuse
the cached assets.

On VRAM-constrained GPUs, add ``--offload-text-encoder`` to reduce peak
VRAM usage by about 15 GB:

.. code-block:: bash

   uv run --package flashdreams-omnidreams interactive-drive \
       --stream-mjpeg :8080 \
       --offload-text-encoder

See :doc:`/models/omnidreams` for the offload trade-offs, scripted
generation, scene variants, WebRTC serving, and multi-GPU options.

Where to next
-------------

- :doc:`/models/index`: every shipped model with its CLI slug and the
  command to run it.
- :doc:`/models/omnidreams`: drive a world model in real time with the
  ``interactive-drive`` demo.
- :doc:`/developer_guides/inference_pipeline_overview`: the generation
  loop end to end: KV cache, ring attention, CUDA-graph capture.
- :doc:`/developer_guides/config_system`: the configuration layer
  every method shares.
- :doc:`/developer_guides/new_integration`: adding a new model or
  method as a plugin.
- :doc:`CLI and API Reference </api/index>`: Reference docs for the
  ``flashdreams-run`` CLI and the FlashDreams Python API.
- :doc:`/troubleshooting`: common first-run failures and fixes.
