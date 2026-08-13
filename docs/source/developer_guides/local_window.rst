.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Local-window presentation
=========================

``local-window`` is a shared realtime output mode that pairs a native keyboard
input source with a SlangPy Vulkan presenter. It uses the same
``RealtimeSessionDriver``, ``SessionEdges``, model runtime, and ``StepResult``
boundary as other demo modes.

Install the optional desktop dependency:

.. code-block:: bash

   uv sync --package flashdreams --extra native-window

The host must provide a local display and a working Vulkan driver. Local-window
mode is single-process; it is not supported under multi-rank ``torchrun``.

Synthetic application
---------------------

The triangle app and model integration exercise package discovery,
runtime/session creation, shared realtime driving, finite-tail draining, and
native presentation without a model checkpoint:

.. code-block:: bash

   uv run --package flashdreams-triangle-model \
     flashdreams-run triangle-model

The application generates a moving triangle for six seconds. Frames are queued
in bounded whole chunks, stale chunks are dropped to limit latency, and normal
completion drains the final chunk before the window closes.

Controls and shutdown
---------------------

The shared presenter forwards W/A/S/D, Q/E, I/J/K/L, R/G/B, Space, Shift,
Control, and arrow aliases as ordinary ``key_down`` and ``key_up`` events.
Escape closes the window. Applications decide which events are meaningful.

Window close requests stop through the run mode transport. Cleanup waits are
bounded by ``NativeWindowOutputSpec.close_timeout_s`` so a hung model operation
does not freeze the UI indefinitely.

Manual presenter smoke test
---------------------------

Headless CI validates the input source, queue, output sink, run mode, lifecycle,
and a fake presenter. Run the real display smoke test on a Linux Vulkan host:

.. code-block:: bash

   uv run --package flashdreams --extra dev --extra native-window \
     pytest -m manual flashdreams/tests/test_native_window_manual.py

Current scope
-------------

The generic presenter defaults to batch 0, view 0; manifests may set
``output.batch_index`` and ``output.view_index`` explicitly. Frames currently
use a host NumPy upload. This path does not replace OmniDreams' specialized
local application, which owns physics, scene switching, HUD/BEV composition,
wheel input, and CUDA/Vulkan interop. Those features remain integration-owned.
