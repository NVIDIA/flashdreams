.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Synthetic input files
=====================

MP4 runs can replay deterministic user input from a version 1 JSON file:

.. code-block:: json

   {
      "version": 1,
      "events": [
         {
            "on": "ui_loop",
            "at": 0,
            "event": {"type": "keyboard", "key": "w", "state": "Pressed"}
         }
      ]
   }

Pass the file with ``--synthetic-input-file PATH``. The top-level object must
contain only ``version`` and ``events``. Version 1 supports ``ui_loop``
scheduling. ``at`` is a zero-based input-poll index: ``0`` releases an event
when the runtime first polls the client window after opening the session,
before the first UI step. Events with the same ``at`` value retain their file
order.

Event payloads
--------------

Each ``event`` object must contain a registered ``UserInputEvent`` ``type`` and
may contain only that event type's fields. Do not provide ``timestamp``; the
runtime stamps an event when it is released. Fields listed below as optional
use their API defaults when omitted.

``numeral_keypad``
   Optional ``value`` integer.

``keyboard``
   Required ``key`` string and ``state`` of ``"Pressed"`` or ``"Released"``.

``close``
   No additional fields.

``reset``
   No additional fields.

``mouse``
   Optional ``action`` (``"move"``, ``"button"``, or ``"wheel"``), ``x`` and
   ``y`` numbers, ``button`` integer, ``pressed`` boolean, and ``wheel_x`` and
   ``wheel_y`` numbers.

``focus``
   Optional ``focused`` boolean.

``query_string``
   Optional ``query_string`` string.

``touch``
   Optional ``action`` (``"start"``, ``"move"``, ``"end"``, or ``"cancel"``),
   ``touch_id`` integer, ``x``, ``y``, and ``pressure`` numbers, and ``primary``
   boolean.

``gamepad``
   Optional ``action`` (``"connected"``, ``"disconnected"``, or ``"state"``),
   ``index`` integer, ``controller_id`` and ``mapping`` strings, ``axes`` and
   ``buttons`` number arrays, and ``pressed`` boolean array.

``game_wheel``
   Optional ``action`` (``"connected"``, ``"disconnected"``, or ``"state"``),
   ``index`` integer, ``controller_id`` string, ``steering``, ``throttle``,
   ``brake``, and ``clutch`` numbers, and ``buttons`` boolean array.

``xr_controller``
   Optional ``action`` (``"connected"``, ``"disconnected"``, or ``"state"``),
   ``handedness`` (``"left"``, ``"right"``, or ``"none"``),
   ``controller_id`` string, ``axes`` and ``buttons`` number arrays,
   ``pressed`` boolean array, three-number ``position`` or ``null``, and
   four-number ``orientation`` or ``null``.

``unknown``
   No additional fields.
