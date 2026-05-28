# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Linux ``evdev`` helpers shared by the demo and the wheel configurator.

Lives in its own module (rather than directly in :mod:`demo`) so the
:mod:`omnidreams.interactive_drive.wheel_configurator` CLI can run
without the ``interactive-drive`` extra (slangpy) installed -- importing
``demo`` would otherwise transitively pull in slangpy via
:class:`~omnidreams.interactive_drive.app.InteractiveDriveApp`.
"""

from __future__ import annotations

import array
import fcntl
import struct
from dataclasses import dataclass
from pathlib import Path

EVDEV_EVENT_FORMAT = "llHHi"
EVDEV_EVENT_SIZE = struct.calcsize(EVDEV_EVENT_FORMAT)

EV_ABS = 0x03
EV_FF = 0x15

FF_CONSTANT = 0x52
FF_AUTOCENTER = 0x61
FF_GAIN = 0x60

# ``EVIOCSFF`` -- upload (or update) a ``struct ff_effect`` for playback.
# Hand-coded so we don't depend on ``ioctl_opt`` / evdev for a single
# constant. Layout: _IOC(_IOC_WRITE, 'E', 0x80, sizeof(struct ff_effect))
# where sizeof(ff_effect) = 0x30 on 64-bit Linux and the kernel encodes
# this as 0x40304580.
EVIOCSFF = 0x40304580


def EVIOCGABS(axis: int) -> int:  # noqa: N802 - mirrors the kernel macro name
    """Compute the ``EVIOCGABS(axis)`` ioctl number for one ABS code."""
    return 0x80184540 + axis


# ``EVIOCGNAME(256)``; pre-baked here rather than computed because it's
# the only name-buffer length the demo + configurator use.
_EVIOCGNAME_256 = 0x80004506 + (256 << 16)


@dataclass(frozen=True)
class AxisRange:
    minimum: int
    maximum: int

    @property
    def center(self) -> float:
        return (float(self.minimum) + float(self.maximum)) * 0.5

    @property
    def span(self) -> float:
        return max(1.0, float(self.maximum - self.minimum))


@dataclass(frozen=True)
class AxisInfo:
    """Current value + range for one ABS axis (``input_absinfo`` mirror).

    Used by the wheel configurator to read the axis position *right
    now* without having to stream events. The runtime ``WheelBridge``
    only needs :class:`AxisRange`, which is exposed via :attr:`range`.
    """

    value: int
    minimum: int
    maximum: int

    @property
    def range(self) -> AxisRange:
        return AxisRange(minimum=self.minimum, maximum=self.maximum)


@dataclass(frozen=True)
class EvdevDevice:
    path: Path
    name: str


def query_axis(path: Path, axis: int) -> AxisInfo | None:
    """Query the current value + min/max of one ABS axis.

    Returns ``None`` when the device does not expose the requested
    axis (``EVIOCGABS`` returns ``ENOTTY``/``EINVAL`` in that case).
    """
    try:
        with path.open("rb") as handle:
            payload = array.array("i", [0, 0, 0, 0, 0, 0])
            fcntl.ioctl(handle.fileno(), EVIOCGABS(axis), payload, True)
            return AxisInfo(
                value=int(payload[0]),
                minimum=int(payload[1]),
                maximum=int(payload[2]),
            )
    except OSError:
        return None


def query_axis_range(path: Path, axis: int) -> AxisRange | None:
    """Convenience wrapper that returns just the static min/max."""
    info = query_axis(path, axis)
    return None if info is None else info.range


def read_evdev_name(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            name_buf = array.array("B", [0] * 256)
            fcntl.ioctl(handle.fileno(), _EVIOCGNAME_256, name_buf)
            return name_buf.tobytes().split(b"\x00")[0].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def scan_evdev_devices() -> tuple[EvdevDevice, ...]:
    """Enumerate readable evdev devices, preferring stable ``by-id`` paths.

    Devices under ``/dev/input/by-id`` come first because their paths
    survive USB re-plug and reboot; the raw ``/dev/input/event*`` paths
    follow as a fallback for devices that don't expose a ``by-id``
    symlink (e.g. virtual/uinput sources).
    """
    candidates: list[Path] = []
    by_id = Path("/dev/input/by-id")
    if by_id.is_dir():
        candidates.extend(
            sorted(path for path in by_id.glob("*event*") if path.exists())
        )
    candidates.extend(sorted(Path("/dev/input").glob("event*")))

    devices: list[EvdevDevice] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        name = read_evdev_name(path)
        if name is not None:
            devices.append(EvdevDevice(path=path, name=name))
    return tuple(devices)
