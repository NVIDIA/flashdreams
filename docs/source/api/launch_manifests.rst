.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Launch manifests
================

Every FlashDreams model launch starts with the same command shape:

.. code-block:: bash

   uv run flashdreams-run <runner-slug> [mode] [--manifest PATH]

``run`` is the default mode. Integrations may additionally expose ``mp4``,
``null``, ``webrtc``, or ``local-window``. Inspect a resolved launch without
loading checkpoints or initializing CUDA with ``--no-instantiate``.

Schema
------

Launch manifests are strict, versioned YAML documents:

.. code-block:: yaml

   schema_version: 1
   runner: omnidreams
   mode: webrtc

   runner_overrides:
     device: cuda:0

   scenario:
     scene_uuid: 0d404ff7-2b66-498c-b047-1ed8cded60d4
     scene_variant: default

   output:
     host: 0.0.0.0
     port: 8089

``schema_version``, ``runner``, and ``mode`` are required. The remaining
sections are optional mappings:

.. note::

   Quote the null-output mode as ``mode: "null"`` in YAML; an unquoted
   ``null`` is YAML's null scalar rather than the FlashDreams mode name.

``runner_overrides``
   Recursive overrides for the registered runner configuration. The same
   runner fields remain available as explicit CLI flags.

``scenario``
   Inputs and controls such as prompts, example data, scenes, traces, and
   rollout length. The selected integration validates the accepted fields.

``output``
   Transport or artifact settings such as output path, frame rate, WebRTC
   bind address, warmup, and local-window presentation settings.

Relative paths in keys named ``path`` or ending in ``_path``, ``_paths``, or
``_dir`` resolve relative to the manifest file, which makes checked-in launch
manifests reproducible from any working directory. Unknown top-level or
integration-specific fields fail before CUDA initialization.

Precedence
----------

Settings resolve in this order, from lowest to highest precedence:

.. code-block:: text

   registered runner preset
     < manifest runner_overrides
     < manifest scenario/output
     < explicit CLI runner flags and --host/--port

The runner and positional mode must agree with the manifest. For example, this
fails instead of silently launching a different preset:

.. code-block:: bash

   uv run flashdreams-run lingbot-world-fast mp4 \
       --manifest configs/launch_manifest/lingbot_webrtc.yaml

Examples
--------

.. code-block:: bash

   # WebRTC
   uv run flashdreams-run lingbot-world-fast webrtc \
       --manifest configs/launch_manifest/lingbot_webrtc.yaml

   # MP4 replay
   uv run flashdreams-run lingbot-world-fast mp4 \
       --manifest configs/launch_manifest/lingbot_mp4.yaml

   # Resolve an OmniDreams launch without loading the model
   uv run flashdreams-run \
       omnidreams webrtc \
       --manifest configs/launch_manifest/omnidreams_webrtc.yaml \
       --no-instantiate

OmniDreams local-window also accepts the existing
``example_world_model*.yaml`` format directly as a compatibility input. New
automation should use a versioned launch manifest whose
``output.world_model_manifest_path`` references that model-specific file.
