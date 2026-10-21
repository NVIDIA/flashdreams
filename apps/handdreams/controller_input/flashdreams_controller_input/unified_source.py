# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exclusive controller ownership across relayed and server-local sources."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from flashdreams_controller_input.controller_types import InputSource, Observation
from flashdreams_controller_input.events import ControllerConnectionStatus
from flashdreams_controller_input.frames import SIDES, both_controllers_tracked
from flashdreams_controller_input.isaac_teleop_source import IsaacTeleopSource

if TYPE_CHECKING:
    from flashdreams_controller_input.quest_webxr import QuestWebXRProducer

_MAX_SAMPLE_AGE_NS = 500_000_000
"""Match the underlying sources' half-second host-observation freshness limit."""

_MAX_DRAIN_OBSERVATIONS = 256
"""Bound each drain to the remote source's maximum pending-frame count."""


class UnifiedControllerSource:
    """One active producer, acquired by a fresh tracked controller pair.

    Either tracked hand renews ownership. A second of complete tracking loss or
    silence releases it; reacquisition resets the origin generation. Inactive
    samples are discarded, never saved for a later handover. Simultaneous
    candidates are ordered by host capture time, with remote winning exact ties.
    """

    name = "unified-controllers"

    def __init__(
        self,
        remote: InputSource,
        local: IsaacTeleopSource,
        *,
        headset: QuestWebXRProducer | None = None,
        web_client_url: str = "",
        certificate_url: str = "",
        ownership_timeout_s: float = 1.0,
        _clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        """Keep existing source sampling and require a bounded ownership lease."""
        if not math.isfinite(ownership_timeout_s) or ownership_timeout_s <= 0:
            raise ValueError("controller ownership timeout must be finite and positive")
        self._remote = remote
        self._local = local
        self._sources = (remote, local)
        self._headset = headset
        self._web_client_url = web_client_url
        self._certificate_url = certificate_url
        self._timeout_ns = int(ownership_timeout_s * 1e9)
        self._clock_ns = _clock_ns
        self._stopped = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._starting = False
        self._started = False
        self._owner: int | None = None
        self._owner_source = ""
        self._owner_generation = 0
        self._generation = -1
        self._last_tracking_ns = 0
        self._acquire_after_ns = 0

    def start(self) -> None:
        """Start the remote listener before launching local input without a browser wait."""
        with self._lifecycle_lock:
            if self._stopped.is_set():
                raise RuntimeError(
                    "unified controller source was stopped before starting"
                )
            if self._started:
                return
            if self._starting:
                raise RuntimeError("unified controller source is already starting")
            self._starting = True
        try:
            self._remote.start()
            if self._stopped.is_set():
                raise RuntimeError("unified controller source stopped during startup")
            self._local.start(wait_for_frame=False)
            if self._headset is not None and not self._stopped.is_set():
                self._headset.start()
            with self._lifecycle_lock:
                if self._stopped.is_set():
                    raise RuntimeError(
                        "unified controller source stopped during startup"
                    )
                self._started = True
        except BaseException:
            # Repeat cleanup after a cancelled start returns: it may own a late child.
            self.stop()
            raise
        finally:
            with self._lifecycle_lock:
                self._starting = False

    def stop(self) -> None:
        """Cancel startup and stop both idempotent sources, even if one cleanup fails."""
        self._stopped.set()
        with self._stop_lock:
            try:
                if self._headset is not None:
                    self._headset.stop()
            finally:
                try:
                    self._local.stop()
                finally:
                    self._remote.stop()

    def connection_status(self) -> ControllerConnectionStatus:
        """Report startup guidance without consuming frames or probing devices."""
        ready = self._local.ready_for_clients and not self._stopped.is_set()
        return ControllerConnectionStatus(
            headset=self._headset.status_message if self._headset is not None else "",
            web_client="Ready to connect" if ready else "Starting...",
            web_client_url=self._web_client_url if ready else "",
            certificate_url=self._certificate_url if ready else "",
            active_input=(
                "Waiting for controllers"
                if self._owner is None
                else "Remote headset"
                if self._owner == 0
                else "Isaac Teleop"
            ),
        )

    def can_connect_headset(self) -> bool:
        """Keep automatic Quest connection from replacing the current input owner."""
        return self._owner is None and not self._stopped.is_set()

    def _fresh(self, observation: Observation) -> bool:
        return 0 <= self._clock_ns() - observation.capture_ns <= _MAX_SAMPLE_AGE_NS

    def _drain(self, index: int) -> None:
        # ponytail: bound work for a source that never returns None; increase this
        # with the transport queue limit if it ever exceeds 256 observations.
        for _ in range(_MAX_DRAIN_OBSERVATIONS):
            if self._stopped.is_set() or self._sources[index].poll() is None:
                break

    def _candidate(self, index: int) -> Observation | None:
        for _ in range(_MAX_DRAIN_OBSERVATIONS):
            if self._stopped.is_set():
                return None
            observation = self._sources[index].poll()
            if observation is None:
                return None
            if (
                observation.capture_ns >= self._acquire_after_ns
                and self._fresh(observation)
                and both_controllers_tracked(observation.frame)
            ):
                return observation
        return None

    def _acquire(self) -> Observation | None:
        candidates = [
            (observation.capture_ns, index, observation)
            for index in range(2)
            if (observation := self._candidate(index)) is not None
        ]
        if not candidates:
            return None
        _, index, observation = min(candidates, key=lambda item: item[:2])
        self._drain(1 - index)
        self._owner = index
        # A direct CloudXR connection does not identify a browser as IWER.
        self._owner_source = observation.source
        self._owner_generation = observation.generation
        self._generation += 1
        self._last_tracking_ns = observation.capture_ns
        return self._selected(observation)

    def _selected(self, observation: Observation) -> Observation:
        return replace(
            observation, source=self._owner_source, generation=self._generation
        )

    def poll(self) -> Observation | None:
        """Return only the owner's original sample or a loss boundary, without waiting."""
        if not self._started or self._stopped.is_set():
            return None
        try:
            if self._owner is None:
                result = self._acquire()
            else:
                now = self._clock_ns()
                if now - self._last_tracking_ns >= self._timeout_ns:
                    result = Observation(
                        None, now, self._owner_source, self._generation
                    )
                    self._owner = None
                    # A paced source may hide a buffered frame from the drain.
                    # Only observations captured after release can acquire it.
                    self._acquire_after_ns = now
                    self._drain(0)
                    self._drain(1)
                else:
                    observation = self._sources[self._owner].poll()
                    self._drain(1 - self._owner)
                    result = None
                    if observation is not None:
                        if observation.generation != self._owner_generation:
                            self._owner_generation = observation.generation
                            self._generation += 1
                        if observation.frame is not None and not self._fresh(
                            observation
                        ):
                            result = Observation(
                                None, now, self._owner_source, self._generation
                            )
                        else:
                            if observation.frame is not None and any(
                                isinstance(
                                    controller := observation.frame.get(side), dict
                                )
                                and controller.get("tracked", True)
                                for side in SIDES
                            ):
                                self._last_tracking_ns = observation.capture_ns
                            result = self._selected(observation)
            return None if self._stopped.is_set() else result
        except BaseException:
            self.stop()
            raise
