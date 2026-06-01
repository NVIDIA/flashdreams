# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Device-agnostic evdev capture + calibration inference.

This module has no GUI dependency so it can be unit-tested without
hardware. It works identically for a steering wheel (pedals that rest at
their axis maximum) and a game controller (analog stick + triggers that
rest at zero) -- the difference is captured, not coded.

Capture model: the UI starts a *listening* window (``reset_observed``),
the user operates a control, and the session keeps a running min/max
(peak-hold) per axis plus the first significant deflection. Reading the
result on stop -- rather than snapshotting after the user lets go -- is
what makes self-centering sticks and force-feedback wheels work: the
extreme is retained even though the control springs back before the user
can click.
"""

from __future__ import annotations

import os
import select
import struct
import threading
import time
from pathlib import Path

from omnidreams.interactive_drive.input.wheel_profiles import (
    EV_ABS,
    EV_KEY,
    EVDEV_EVENT_FORMAT,
    EVDEV_EVENT_SIZE,
    AxisRange,
    WheelProfile,
    read_axis_states,
)

# An axis must move at least this fraction of its full range to count as
# "the control the user operated". Keeps idle jitter / unrelated axes from
# being mistaken for the intended one.
MIN_MOVE_FRACTION = 0.15
# A deflection this far from the resting value (as a fraction of full
# range) is recorded as the "first significant" move, used to tell which
# physical direction the user pushed first (for steering invert).
FIRST_DEFLECTION_FRACTION = 0.20


def select_axis_by_span(
    observed: dict[int, AxisRange],
    axis_ranges: dict[int, AxisRange],
    *,
    min_fraction: float = MIN_MOVE_FRACTION,
) -> int | None:
    """Return the axis that moved the most during a listening window.

    Movement is the observed span as a fraction of the axis' full device
    range, so axes with different raw scales (an 8-bit trigger vs a 16-bit
    wheel) compare fairly. ``None`` if nothing moved beyond *min_fraction*.
    """
    best_code: int | None = None
    best_fraction = 0.0
    for code, observed_range in observed.items():
        full = axis_ranges.get(code)
        span = full.span if full is not None else 65535.0
        fraction = observed_range.span / span
        if fraction > best_fraction:
            best_fraction = fraction
            best_code = code
    return best_code if best_fraction >= min_fraction else None


def peak_from_observed(observed_range: AxisRange, reference: float) -> int:
    """Return the observed extreme farthest from *reference*.

    For a pedal/trigger this is the fully-engaged value relative to its
    resting baseline; for steering it is the held extreme relative to the
    axis center.
    """
    low, high = observed_range.minimum, observed_range.maximum
    return low if abs(low - reference) >= abs(high - reference) else high


def detect_moved_axis(
    before: dict[int, int],
    after: dict[int, int],
    axis_ranges: dict[int, AxisRange],
    *,
    min_fraction: float = MIN_MOVE_FRACTION,
) -> int | None:
    """Return the axis with the largest fractional change between two samples.

    Snapshot-based helper retained for completeness/tests; the wizard uses
    the peak-hold flow (:func:`select_axis_by_span`) instead.
    """
    best_code: int | None = None
    best_fraction = 0.0
    for code in set(before) | set(after):
        if code not in before or code not in after:
            continue
        rng = axis_ranges.get(code)
        span = rng.span if rng is not None else 65535.0
        fraction = abs(after[code] - before[code]) / span
        if fraction > best_fraction:
            best_fraction = fraction
            best_code = code
    return best_code if best_fraction >= min_fraction else None


def infer_steering_invert(left_raw: float, right_raw: float) -> bool:
    """Whether steering must be inverted so full-left reads as positive steer.

    Pass the full-left raw value and either the full-right raw value or the
    axis center as the reference: if full-left is the lower value the sign
    must be flipped (``command_from_snapshot`` treats positive steer as
    left).
    """
    return left_raw < right_raw


def infer_pedal_inverted(rest_raw: int, pressed_raw: int) -> bool:
    """Whether a pedal/trigger rests high and falls when engaged.

    ``True`` for wheel pedals (rest at axis max, pressed toward min);
    ``False`` for controller triggers (rest at 0, pressed toward max).
    """
    return pressed_raw < rest_raw


def build_profile(
    *,
    name: str,
    display_name: str,
    detection_patterns: tuple[str, ...],
    steering_axis: int,
    throttle_axis: int,
    brake_axis: int,
    invert_steering: bool,
    inverted_pedals: bool,
    ffb_enabled: bool,
    ffb_gain: float,
    is_default: bool,
    reverse_buttons: tuple[int, ...] = (),
    reset_buttons: tuple[int, ...] = (),
    steering_range: float = 1.0,
    steering_deadzone: float = 0.0,
    threshold: float = 0.12,
) -> WheelProfile:
    """Assemble captured calibration values into a :class:`WheelProfile`."""
    return WheelProfile(
        name=name,
        display_name=display_name,
        detection_patterns=tuple(detection_patterns),
        axis_map={
            "steering": int(steering_axis),
            "throttle": int(throttle_axis),
            "brake": int(brake_axis),
        },
        inverted_pedals=bool(inverted_pedals),
        invert_steering=bool(invert_steering),
        ffb_enabled=bool(ffb_enabled),
        ffb_gain=float(ffb_gain),
        threshold=float(threshold),
        is_default=bool(is_default),
        reverse_buttons=tuple(int(b) for b in reverse_buttons),
        reset_buttons=tuple(int(b) for b in reset_buttons),
        steering_range=float(steering_range),
        steering_deadzone=float(steering_deadzone),
    )


class CaptureSession:
    """Background reader for a single evdev device.

    On construction it seeds the current value and min/max range of every
    absolute axis from ``EVIOCGABS`` so the UI can show all inputs live
    immediately. During a listening window it keeps a running min/max
    (peak-hold) and the first significant deflection per axis.
    """

    def __init__(self, device_path: Path) -> None:
        self._device_path = Path(device_path)
        self._lock = threading.Lock()
        states = read_axis_states(self._device_path)
        self._axis_ranges: dict[int, AxisRange] = {
            code: rng for code, (_value, rng) in states.items()
        }
        self._raw_axes: dict[int, int] = {
            code: value for code, (value, _rng) in states.items()
        }
        self._raw_buttons: dict[int, int] = {}
        self._pressed_buttons: set[int] = set()
        self._observed: dict[int, list[int]] = {}
        self._baseline: dict[int, int] = dict(self._raw_axes)
        self._first_big: dict[int, int] = {}
        self._last_event_t = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None

    @property
    def device_path(self) -> Path:
        return self._device_path

    @property
    def axis_ranges(self) -> dict[int, AxisRange]:
        return dict(self._axis_ranges)

    def start(self) -> None:
        """Open the device and start the reader thread.

        Raises ``OSError`` (e.g. ``PermissionError``) if the device cannot
        be opened, so the caller can show actionable guidance.
        """
        self._fd = os.open(self._device_path, os.O_RDONLY | os.O_NONBLOCK)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="input-config-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _run(self) -> None:
        fd = self._fd
        if fd is None:
            return
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([fd], [], [], 0.02)
            except OSError:
                break
            if readable:
                self._read_events(fd)

    def _read_events(self, fd: int) -> None:
        try:
            data = os.read(fd, EVDEV_EVENT_SIZE * 64)
        except (BlockingIOError, OSError):
            return
        now = time.monotonic()
        with self._lock:
            for offset in range(0, len(data) - EVDEV_EVENT_SIZE + 1, EVDEV_EVENT_SIZE):
                _, _, event_type, code, value = struct.unpack(
                    EVDEV_EVENT_FORMAT, data[offset : offset + EVDEV_EVENT_SIZE]
                )
                code = int(code)
                value = int(value)
                if event_type == EV_ABS:
                    self._raw_axes[code] = value
                    bounds = self._observed.get(code)
                    if bounds is None:
                        self._observed[code] = [value, value]
                    else:
                        bounds[0] = min(bounds[0], value)
                        bounds[1] = max(bounds[1], value)
                    if code not in self._first_big:
                        rng = self._axis_ranges.get(code)
                        span = rng.span if rng is not None else 65535.0
                        base = self._baseline.get(code, value)
                        if abs(value - base) > FIRST_DEFLECTION_FRACTION * span:
                            self._first_big[code] = value
                    self._last_event_t = now
                elif event_type == EV_KEY:
                    self._raw_buttons[code] = value
                    if value == 1:
                        self._pressed_buttons.add(code)
                    self._last_event_t = now

    def axes(self) -> dict[int, int]:
        with self._lock:
            return dict(self._raw_axes)

    def buttons(self) -> dict[int, int]:
        with self._lock:
            return dict(self._raw_buttons)

    def reset_observed(self) -> None:
        """Begin a fresh listening window from the current resting values."""
        with self._lock:
            self._observed = {}
            self._first_big = {}
            self._pressed_buttons = set()
            self._baseline = dict(self._raw_axes)

    def pressed_buttons(self) -> set[int]:
        """Button codes (EV_KEY) pressed since the last :meth:`reset_observed`."""
        with self._lock:
            return set(self._pressed_buttons)

    def observed_ranges(self) -> dict[int, AxisRange]:
        with self._lock:
            return {
                code: AxisRange(minimum=bounds[0], maximum=bounds[1])
                for code, bounds in self._observed.items()
            }

    def baseline(self) -> dict[int, int]:
        """Resting values captured at the start of the listening window."""
        with self._lock:
            return dict(self._baseline)

    def first_big(self) -> dict[int, int]:
        """First significant deflection value per axis during the window."""
        with self._lock:
            return dict(self._first_big)

    def is_active(self, window_s: float = 0.4) -> bool:
        with self._lock:
            if not self._last_event_t:
                return False
            return (time.monotonic() - self._last_event_t) < window_s
