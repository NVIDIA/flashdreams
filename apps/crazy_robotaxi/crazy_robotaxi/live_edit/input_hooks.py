# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rising-edge key requests for the live-edit abilities.

Mirrors the one-slot request/consume channels on
:class:`crazy_robotaxi.input.CrazyRobotaxiKeyboardState`
(``submit_taxi_name`` / ``consume_taxi_name_submission``). Kept as a
separate object so the presenter key handlers and the runtime drain can
share it without subclassing the keyboard state; composition-root wiring:

- native window: add ``k``/``c`` keysyms to ``_build_key_codes`` and call
  ``requests.request_skin_cycle()`` / ``request_coins_toggle()`` from the
  discrete tail of ``SlangPyHudPresenter._on_keyboard_event``;
- MJPEG: same calls from ``MJPEGStreamingPresenter._apply_control`` plus the
  browser JS key allowlist;
- drain: ``CrazyRobotaxiRuntime.process_events`` consumes both each tick.
"""

from __future__ import annotations

import threading


class LiveEditRequests:
    """Thread-safe one-shot requests raised by input, drained by the runtime."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._skin_cycle_requested = False
        self._coins_toggle_requested = False
        self._weather_cycle_requested = False
        self._obstacle_spawn_requested = False

    def request_skin_cycle(self) -> None:
        """Record a switch-skin keypress until the runtime consumes it."""
        with self._lock:
            self._skin_cycle_requested = True

    def consume_skin_cycle(self) -> bool:
        """Return and clear whether a skin switch was requested."""
        with self._lock:
            requested = self._skin_cycle_requested
            self._skin_cycle_requested = False
            return requested

    def request_coins_toggle(self) -> None:
        """Record a coins-toggle keypress until the runtime consumes it."""
        with self._lock:
            self._coins_toggle_requested = True

    def consume_coins_toggle(self) -> bool:
        """Return and clear whether a coins toggle was requested."""
        with self._lock:
            requested = self._coins_toggle_requested
            self._coins_toggle_requested = False
            return requested

    def request_weather_cycle(self) -> None:
        """Record a cycle-weather keypress until the runtime consumes it."""
        with self._lock:
            self._weather_cycle_requested = True

    def consume_weather_cycle(self) -> bool:
        """Return and clear whether a weather cycle was requested."""
        with self._lock:
            requested = self._weather_cycle_requested
            self._weather_cycle_requested = False
            return requested

    def request_obstacle_spawn(self) -> None:
        """Record a spawn-obstacle keypress until the runtime consumes it."""
        with self._lock:
            self._obstacle_spawn_requested = True

    def consume_obstacle_spawn(self) -> bool:
        """Return and clear whether an obstacle spawn was requested."""
        with self._lock:
            requested = self._obstacle_spawn_requested
            self._obstacle_spawn_requested = False
            return requested
