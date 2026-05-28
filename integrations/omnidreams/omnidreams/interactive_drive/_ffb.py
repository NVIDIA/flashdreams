# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Force-feedback backends shared by the demo + the wheel configurator.

Two strategies are supported, selected per-profile by ``WheelProfile.ffb_mode``:

* ``"autocenter"`` -- :class:`AutocenterFFB`: writes ``FF_AUTOCENTER`` +
  ``FF_GAIN`` events, letting the kernel/firmware shape the centering
  curve. Works on Thrustmaster, Logitech, and any driver that wires up
  the in-kernel autocenter handler. Does **not** work on Fanatec --
  ``hid-fanatecff`` accepts the write but never acts on it -- so any
  Fanatec setup configured with ``mode: autocenter`` will appear to
  have FFB enabled while producing no force at all.
* ``"constant_force"`` -- :class:`ConstantForceFFB`: uploads a single
  ``FF_CONSTANT`` effect via ``EVIOCSFF`` and re-uploads new level
  values every tick. Works on Fanatec, Simagic, Moza, *and*
  Thrustmaster, at the cost of computing the centering curve in
  userspace (sqrt-of-displacement, speed-scaled).

For back-compat, profiles without an ``ffb.mode`` key default to
``"autocenter"`` (matches the original single-backend behaviour). The
wheel configurator probes ``constant_force`` first during calibration
and falls back to ``autocenter`` only if the user reports no force.
"""

from __future__ import annotations

import fcntl
import os
import struct
import time
from pathlib import Path

from omnidreams.interactive_drive._evdev import (
    EV_FF,
    EVDEV_EVENT_FORMAT,
    EVIOCSFF,
    FF_AUTOCENTER,
    FF_CONSTANT,
    FF_GAIN,
)

# ``struct ff_effect`` size on 64-bit Linux. Matches the kernel layout
# referenced by ``EVIOCSFF``: u16 type, s16 id, u16 direction,
# u16+u16 trigger, u16+u16 replay, 2-byte alignment pad, then the
# union (constant.level lives at offset 16).
_FF_EFFECT_STRUCT_SIZE = 48

# Direction encoding: a quarter-circle field where 0x0000=north,
# 0x4000=east, 0x8000=south, 0xC000=west. For a single-axis steering
# wheel the direction is effectively ignored -- the sign of
# ``constant.level`` selects which way the motor pushes -- but the
# kernel still validates the field, so we pick a fixed value.
_FF_DIRECTION_EAST = 0x4000


class AutocenterFFB:
    """In-kernel autocenter; the original single FFB path the demo shipped.

    Sets a speed-dependent ``FF_AUTOCENTER`` strength every few ticks.
    The kernel / wheel firmware shapes the actual centering curve, so
    feel varies between hardware vendors.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._last_strength = -1
        self._smoothed = 0.0

    def init(self, device_path: Path, gain: float) -> None:
        try:
            self._fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
            self._write_event(FF_AUTOCENTER, 0)
            self._write_event(FF_GAIN, int(max(0.0, min(1.0, gain)) * 0xFFFF))
            print(f"[demo] FFB autocenter enabled on {device_path}", flush=True)
        except PermissionError:
            print(
                "[demo] FFB permission denied; add user to input group or adjust udev",
                flush=True,
            )
            self._fd = None
        except OSError as exc:
            print(f"[demo] FFB unavailable on {device_path}: {exc}", flush=True)
            self._fd = None

    def update(
        self,
        speed_mps: float,
        *,
        gain: float,
        steering_raw: int,  # noqa: ARG002 - kept for backend-uniform signature
        steering_center: float,  # noqa: ARG002
        steering_span: float,  # noqa: ARG002
    ) -> None:
        if self._fd is None:
            return
        if speed_mps < 0.1:
            target = 0.15
        else:
            norm = min(1.0, speed_mps / 14.0)
            target = 0.35 + 0.65 * norm
        self._smoothed += 0.12 * (target - self._smoothed)
        strength = int(self._smoothed * max(0.0, min(1.0, gain)) * 0xFFFF)
        strength = max(0, min(0xFFFF, strength))
        if abs(strength - self._last_strength) > 500:
            self._write_event(FF_AUTOCENTER, strength)
            self._last_strength = strength

    def cleanup(self) -> None:
        if self._fd is None:
            return
        try:
            self._write_event(FF_AUTOCENTER, 0)
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None

    def _write_event(self, code: int, value: int) -> None:
        if self._fd is None:
            return
        now = time.time()
        sec = int(now)
        usec = int((now - sec) * 1_000_000)
        try:
            os.write(
                self._fd,
                struct.pack(EVDEV_EVENT_FORMAT, sec, usec, EV_FF, code, value),
            )
        except OSError:
            return


class ConstantForceFFB:
    """FFB via FF_CONSTANT effect (Fanatec, Simagic, Moza, Thrustmaster).

    Uploads one ``FF_CONSTANT`` effect at init time, plays it
    continuously, and re-uploads new level values every tick. The
    centering force is ``sign(displacement) * sqrt(|displacement|)``
    where ``displacement`` is the normalised offset of the steering
    raw value from the axis centre, scaled by a low-pass-filtered
    speed factor and the profile's ``ffb_gain``.

    Sign convention (mirrors the alpasim port we ported this from):
    a positive ``level`` is taken to push the wheel in the *opposite*
    direction to ``displacement > 0``, i.e. a restoring torque when
    the raw value sits above the axis centre. This holds on every
    Linux-supported wheel base we've tested; if a future device
    reverses the convention and the wheel "runs away" rather than
    centering, hand-flip the sign in the ``level`` line below (or add
    a profile field for it).
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._effect_id: int = -1
        self._last_level: int = 0
        self._smoothed: float = 0.0

    def init(self, device_path: Path, gain: float) -> None:
        try:
            self._fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
        except PermissionError:
            print(
                "[demo] FFB permission denied; add user to input group or adjust udev",
                flush=True,
            )
            self._fd = None
            return
        except OSError as exc:
            print(f"[demo] FFB unavailable on {device_path}: {exc}", flush=True)
            self._fd = None
            return

        # ``FF_GAIN`` is a separate, evdev-wide knob -- not specific to
        # the constant effect -- so we still set it here so subsequent
        # writes scale to the profile's gain ceiling.
        self._write_event(FF_GAIN, int(max(0.0, min(1.0, gain)) * 0xFFFF))

        self._effect_id = -1
        effect_id = self._upload_constant(level=1)
        if effect_id < 0:
            print(
                "[demo] FFB constant-force upload failed; driver likely "
                "lacks FF_CONSTANT support",
                flush=True,
            )
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            return
        self._effect_id = effect_id
        self._play(effect_id, 1)
        self._upload_constant(level=0)
        print(f"[demo] FFB constant-force enabled on {device_path}", flush=True)

    def update(
        self,
        speed_mps: float,
        *,
        gain: float,
        steering_raw: int,
        steering_center: float,
        steering_span: float,
    ) -> None:
        if self._fd is None or self._effect_id < 0:
            return
        if speed_mps < 0.1:
            target = 0.0
        else:
            target = 0.25 + 0.75 * min(1.0, speed_mps / 13.9)
        self._smoothed += 0.15 * (target - self._smoothed)

        half_span = max(1.0, steering_span * 0.5)
        displacement = (float(steering_raw) - steering_center) / half_span
        # Stronger near centre to keep the wheel from feeling limp at
        # small angles.
        sign = 1.0 if displacement >= 0 else -1.0
        shaped = sign * (abs(displacement) ** 0.5)
        force = shaped * self._smoothed * max(0.0, gain)
        level = int(force * 0x7FFF)
        level = max(-0x7FFF, min(0x7FFF, level))
        # Hysteresis: avoid spamming EVIOCSFF on every sub-LSB change;
        # 100 lsb out of 0x7FFF (~0.3%) is below the wheel's mechanical
        # discrimination.
        if abs(level - self._last_level) > 100:
            self._upload_constant(level=level)
            self._last_level = level

    def cleanup(self) -> None:
        if self._fd is None:
            return
        try:
            if self._effect_id >= 0:
                self._upload_constant(level=0)
                self._play(self._effect_id, 0)
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None
        self._effect_id = -1

    def _upload_constant(self, *, level: int) -> int:
        if self._fd is None:
            return -1
        buf = bytearray(_FF_EFFECT_STRUCT_SIZE)
        struct.pack_into("Hh", buf, 0, FF_CONSTANT, self._effect_id)
        struct.pack_into("H", buf, 4, _FF_DIRECTION_EAST)
        struct.pack_into("h", buf, 16, max(-0x7FFF, min(0x7FFF, level)))
        try:
            fcntl.ioctl(self._fd, EVIOCSFF, buf)
        except OSError:
            return -1
        # The kernel writes the assigned effect id back into the struct
        # when ``id == -1`` on entry (i.e. on the first upload). Read it
        # back so subsequent re-uploads address the same effect.
        result_id = struct.unpack_from("Hh", buf, 0)[1]
        self._effect_id = result_id
        return result_id

    def _play(self, effect_id: int, value: int) -> None:
        if self._fd is None:
            return
        self._write_event(effect_id, value)

    def _write_event(self, code: int, value: int) -> None:
        if self._fd is None:
            return
        now = time.time()
        sec = int(now)
        usec = int((now - sec) * 1_000_000)
        try:
            os.write(
                self._fd,
                struct.pack(EVDEV_EVENT_FORMAT, sec, usec, EV_FF, code, value),
            )
        except OSError:
            return


# Module-level dispatch table so the configurator + demo + future
# backends can extend the mode set without touching the call sites.
_FFB_BACKENDS: dict[str, type] = {
    "autocenter": AutocenterFFB,
    "constant_force": ConstantForceFFB,
}

FFB_MODES: tuple[str, ...] = tuple(_FFB_BACKENDS.keys())


def create_ffb_backend(mode: str) -> AutocenterFFB | ConstantForceFFB:
    """Instantiate the FFB backend selected by a profile's ``ffb.mode``.

    Falls back to :class:`AutocenterFFB` for unknown modes so a typo
    in a hand-edited YAML never deadlocks the demo's startup.
    """
    cls = _FFB_BACKENDS.get(mode)
    if cls is None:
        print(
            f"[demo] unknown FFB mode {mode!r}; falling back to 'autocenter'",
            flush=True,
        )
        cls = AutocenterFFB
    return cls()
