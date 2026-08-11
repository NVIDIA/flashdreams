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

CLI
===================================

FlashDreams exposes a unified command line entry point:
``flashdreams-run``.

Core commands
-------------

List all available runner slugs:

.. code-block:: bash

   uv run flashdreams-run --help

Inspect one runner's full options:

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-taehv --help

Run a single-GPU inference (``run`` is the default mode):

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-taehv --total-blocks 7

Launch a WebRTC demo from a versioned manifest:

.. code-block:: bash

   uv run flashdreams-run lingbot-world-fast webrtc \
       --manifest configs/launch_manifest/lingbot_webrtc.yaml

The common command shape is ``flashdreams-run <runner> [mode]``. A runner only
advertises modes it implements; unsupported pairs fail before CUDA
initialization. Shared modes are ``run``, ``mp4``, ``null``, ``webrtc``, and
``local-window``.

Run a multi-GPU inference:

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b-taehv --total-blocks 7

Resolve config only (no model instantiation):

.. code-block:: bash

   uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-taehv --no-instantiate

Post-processing presets
-----------------------

Post-processing presets run on decoded RGB frames from a video runner. Select
one with ``--postprocess.preset``:

.. code-block:: bash

   uv run flashdreams-run wan21-t2v-1.3b-480p \
       --postprocess.preset rtx-super-resolution

The ``rtx-super-resolution`` preset wraps NVIDIA VFX Python bindings for RTX
Video Super Resolution. Install the optional dependency with
``uv pip install 'flashdreams[rtx-postprocess]'`` and run on a supported RTX GPU
before selecting this preset.

See also
--------

- :doc:`/quickstart/index`
- :doc:`/api/launch_manifests`
- :doc:`/developer_guides/config_system`
- :doc:`/developer_guides/runner_slugs`
- :doc:`/api/infra`
