# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared input-device profile model, evdev helpers, and profile IO.

A *profile* describes any analog driving input device -- a steering wheel
with pedals or a game controller with an analog stick and triggers. The
runtime (:class:`~omnidreams.interactive_drive.demo.WheelBridge`) only ever
reads three absolute axes (steering / throttle / brake) and normalizes
them, so the same profile shape covers both device classes; a controller is
just a profile whose throttle/brake axes are triggers (``pedal.inverted:
false``) and whose force feedback is disabled.

Profiles are data-only YAML. Nothing here hardcodes a device's make or
model: detection is by the device's own evdev-reported name, which the
configuration tool captures live and writes only into the user's local
profile -- never into shipped source or configs.

This module is intentionally free of any GUI / slangpy imports so the
configuration tool (:mod:`omnidreams.interactive_drive.input_config`) and
the demo runtime can both depend on it cheaply.
"""

from __future__ import annotations

import array
import fcntl
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import yaml

# --- evdev wire format / ioctl constants -------------------------------
# Linux input_event struct: two longs (timeval), two u16, one s32.
EVDEV_EVENT_FORMAT = "llHHi"
EVDEV_EVENT_SIZE = struct.calcsize(EVDEV_EVENT_FORMAT)
EV_ABS = 0x03
EV_KEY = 0x01
EV_FF = 0x15
FF_AUTOCENTER = 0x61
FF_GAIN = 0x60
# EVIOCGABS(axis): read an absolute axis' value/min/max range.
EVIOCGABS = lambda axis: 0x80184540 + axis  # noqa: E731
# EVIOCGNAME(len): read the device's human-readable name.
EVIOCGNAME = lambda length: 0x80004506 + (length << 16)  # noqa: E731


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
class EvdevDevice:
    path: Path
    name: str


@dataclass(frozen=True)
class WheelProfile:
    """A driving-input profile (steering wheel or game controller).

    ``axis_map`` holds the evdev ABS codes for steering / throttle / brake.
    ``inverted_pedals`` is shared by throttle and brake: ``True`` when the
    control rests at its axis maximum and falls toward the minimum when
    engaged (typical of wheel pedals), ``False`` when it rests at the
    minimum and rises when engaged (typical of controller triggers).
    """

    name: str
    display_name: str
    detection_patterns: tuple[str, ...]
    axis_map: dict[str, int]
    inverted_pedals: bool = True
    invert_steering: bool = False
    ffb_enabled: bool = False
    ffb_gain: float = 0.5
    threshold: float = 0.12
    is_default: bool = False
    # evdev button codes (EV_KEY) bound to actions; empty when unbound.
    reverse_buttons: tuple[int, ...] = ()
    reset_buttons: tuple[int, ...] = ()
    # Steering feel: output scale (``< 1`` = less sensitive) and a center
    # deadzone fraction (hides analog-stick drift on game controllers).
    steering_range: float = 1.0
    steering_deadzone: float = 0.0


def apply_steering_curve(
    value: float, *, deadzone: float = 0.0, scale: float = 1.0
) -> float:
    """Shape a normalized steering value in ``[-1, 1]``.

    ``deadzone`` (a fraction of the range) is removed around center and the
    remainder rescaled so motion just past it starts from zero -- this hides
    analog-stick drift. ``scale`` then limits the output magnitude (``< 1``
    makes steering less sensitive). The result stays in ``[-1, 1]``.
    """
    if deadzone > 0.0:
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0.0
        sign = 1.0 if value > 0.0 else -1.0
        value = sign * (magnitude - deadzone) / (1.0 - deadzone)
    return max(-1.0, min(1.0, value * scale))


def name_match_strength(device_name: str, patterns) -> int:
    """Score a device name against a profile's detection patterns.

    ``2`` when the name equals a pattern exactly, ``1`` when a pattern is a
    substring of the name, ``0`` otherwise (case-insensitive). Exact beats
    substring so a profile captured as ``"Wireless Controller"`` binds that
    node rather than a sibling like ``"Wireless Controller Motion Sensors"``
    whose name merely contains the pattern.
    """
    name = device_name.lower()
    lowered = [str(pattern).lower() for pattern in patterns]
    if any(name == pattern for pattern in lowered):
        return 2
    if any(pattern and pattern in name for pattern in lowered):
        return 1
    return 0


# --- evdev device discovery / queries ----------------------------------


def read_evdev_name(path: Path) -> str | None:
    """Return the evdev device name at *path*, or ``None`` if unreadable."""
    try:
        with path.open("rb") as handle:
            name_buf = array.array("B", [0] * 256)
            fcntl.ioctl(handle.fileno(), EVIOCGNAME(256), name_buf)
            return name_buf.tobytes().split(b"\x00")[0].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def query_axis_range(path: Path, axis: int) -> AxisRange | None:
    """Read an absolute axis' [min, max] via ``EVIOCGABS``.

    Returns ``None`` when the device has no such axis (which is also how
    callers test whether a candidate device exposes a required axis).
    """
    try:
        with path.open("rb") as handle:
            payload = array.array("i", [0, 0, 0, 0, 0, 0])
            fcntl.ioctl(handle.fileno(), EVIOCGABS(axis), payload, True)
            return AxisRange(minimum=int(payload[1]), maximum=int(payload[2]))
    except OSError:
        return None


def scan_evdev_devices() -> tuple[EvdevDevice, ...]:
    """Enumerate readable evdev devices.

    Scans ``/dev/input/by-id`` first (stable, descriptive symlinks) then
    ``/dev/input/event*``, de-duplicating by resolved path.
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


def read_axis_states(path: Path) -> dict[int, tuple[int, AxisRange]]:
    """Return ``{abs_code: (current_value, range)}`` for every absolute axis.

    Probes the common ABS code range (0x00-0x3F, which covers sticks,
    triggers, pedals, and hats) via the same ``EVIOCGABS`` ioctl ``evtest``
    uses. That ioctl reports the axis' *current* value alongside its
    min/max, so the configuration tool can seed a live readout and
    normalize movement immediately, without waiting for the device to emit
    its first event.
    """
    states: dict[int, tuple[int, AxisRange]] = {}
    try:
        with path.open("rb") as handle:
            for axis in range(0x40):
                payload = array.array("i", [0, 0, 0, 0, 0, 0])
                try:
                    fcntl.ioctl(handle.fileno(), EVIOCGABS(axis), payload, True)
                except OSError:
                    continue
                value, minimum, maximum = (
                    int(payload[0]),
                    int(payload[1]),
                    int(payload[2]),
                )
                if maximum != minimum:
                    states[axis] = (value, AxisRange(minimum=minimum, maximum=maximum))
    except OSError:
        return {}
    return states


def list_device_axes(path: Path) -> dict[int, AxisRange]:
    """Return the min/max range of every absolute axis the device exposes."""
    return {code: rng for code, (_value, rng) in read_axis_states(path).items()}


# --- profile IO ---------------------------------------------------------


def user_wheel_profiles_dir() -> Path:
    """User-writable directory where generated profiles are stored.

    Resolves to ``$FLASHDREAMS_CACHE_DIR/interactive-drive/wheels`` (the
    same cache convention the scene staging uses). Read on every call so
    tests that monkeypatch :data:`omnidreams.scenes.FLASHDREAMS_CACHE_DIR`
    see the override.
    """
    from omnidreams.scenes import FLASHDREAMS_CACHE_DIR

    return FLASHDREAMS_CACHE_DIR / "interactive-drive" / "wheels"


def _profile_from_data(data: dict, fallback_name: str) -> WheelProfile:
    axis_map = {
        str(key): int(value) for key, value in (data.get("axis_map") or {}).items()
    }
    pedal = data.get("pedal", {}) or {}
    ffb = data.get("ffb", {}) or {}
    return WheelProfile(
        name=str(data.get("name", fallback_name)),
        display_name=str(data.get("display_name", data.get("name", fallback_name))),
        detection_patterns=tuple(
            str(pattern) for pattern in data.get("detection_patterns", ())
        ),
        axis_map=axis_map,
        inverted_pedals=bool(pedal.get("inverted", data.get("inverted_pedals", True))),
        invert_steering=bool(data.get("invert_steering", False)),
        ffb_enabled=bool(ffb.get("enabled", False)),
        ffb_gain=float(ffb.get("gain", 0.5)),
        threshold=float(data.get("threshold", 0.12)),
        is_default=bool(data.get("is_default", False)),
        reverse_buttons=tuple(int(b) for b in data.get("reverse_buttons", ()) or ()),
        reset_buttons=tuple(int(b) for b in data.get("reset_buttons", ()) or ()),
        steering_range=float(data.get("steering_range", 1.0)),
        steering_deadzone=float(data.get("steering_deadzone", 0.0)),
    )


def load_wheel_profile_files(
    profiles_dir: Path,
) -> tuple[tuple[Path, WheelProfile], ...]:
    """Load ``(path, profile)`` for every ``*.yaml`` in *profiles_dir*.

    The configuration tool needs the source path to edit or delete a
    profile in place; the runtime only needs the profiles themselves
    (:func:`load_wheel_profiles`).
    """
    if not profiles_dir.is_dir():
        return tuple()
    entries: list[tuple[Path, WheelProfile]] = []
    for path in sorted(profiles_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries.append((path, _profile_from_data(data, path.stem)))
    return tuple(entries)


def load_wheel_profiles(profiles_dir: Path) -> tuple[WheelProfile, ...]:
    """Load every ``*.yaml`` profile in *profiles_dir* (empty if missing)."""
    return tuple(profile for _path, profile in load_wheel_profile_files(profiles_dir))


def wheel_profile_to_yaml_dict(profile: WheelProfile) -> dict:
    """Serialize a profile to the dict shape :func:`load_wheel_profiles` reads.

    Key order is preserved on dump for human-friendly files; round-tripping
    the result back through :func:`load_wheel_profiles` reproduces an equal
    :class:`WheelProfile`.
    """
    return {
        "name": profile.name,
        "display_name": profile.display_name,
        "is_default": profile.is_default,
        "detection_patterns": list(profile.detection_patterns),
        "axis_map": {
            "steering": int(profile.axis_map["steering"]),
            "throttle": int(profile.axis_map["throttle"]),
            "brake": int(profile.axis_map["brake"]),
        },
        "pedal": {"inverted": profile.inverted_pedals},
        "invert_steering": profile.invert_steering,
        "ffb": {"enabled": profile.ffb_enabled, "gain": profile.ffb_gain},
        "threshold": profile.threshold,
        "reverse_buttons": list(profile.reverse_buttons),
        "reset_buttons": list(profile.reset_buttons),
        "steering_range": profile.steering_range,
        "steering_deadzone": profile.steering_deadzone,
    }


def profile_filename(name: str) -> str:
    """Filesystem-safe ``<slug>.yaml`` filename for a profile *name*."""
    slug = "".join(ch if ch.isalnum() else "-" for ch in name.strip().lower())
    slug = "-".join(part for part in slug.split("-") if part)
    return f"{slug or 'profile'}.yaml"


def _write_profile_yaml(path: Path, profile: WheelProfile) -> None:
    header = (
        "# Generated by interactive-drive-configuration.\n"
        "# Local input-device profile -- not tracked by the repository.\n\n"
    )
    body = yaml.safe_dump(
        wheel_profile_to_yaml_dict(profile), sort_keys=False, default_flow_style=False
    )
    path.write_text(header + body, encoding="utf-8")


def save_wheel_profile(profile: WheelProfile, profiles_dir: Path) -> Path:
    """Write *profile* as a new YAML in *profiles_dir* and return the path."""
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profiles_dir / profile_filename(profile.name)
    _write_profile_yaml(path, profile)
    return path


def update_profile_file(path: Path, profile: WheelProfile) -> None:
    """Rewrite an existing profile file in place (used by the editor)."""
    _write_profile_yaml(path, profile)


def delete_profile_file(path: Path) -> None:
    """Delete a profile file, ignoring a missing file."""
    path.unlink(missing_ok=True)


# --- force feedback -----------------------------------------------------


class AutocenterFFB:
    """Speed-scaled autocenter force feedback via ``FF_AUTOCENTER``.

    Used by the demo runtime for live driving and by the configuration
    tool's FFB test. ``update`` is a no-op until :meth:`init` succeeds, so
    devices without autocenter support (e.g. game controllers) silently do
    nothing.
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
        except PermissionError:
            print(
                "[wheel] FFB permission denied; add user to input group or adjust udev",
                flush=True,
            )
            self._fd = None
        except OSError as exc:
            print(f"[wheel] FFB unavailable on {device_path}: {exc}", flush=True)
            self._fd = None

    @property
    def available(self) -> bool:
        return self._fd is not None

    def update(self, speed_mps: float, *, gain: float) -> None:
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

    def set_autocenter(self, fraction: float) -> None:
        """Directly set the autocenter strength (used by the FFB test slider)."""
        if self._fd is None:
            return
        strength = max(0, min(0xFFFF, int(max(0.0, min(1.0, fraction)) * 0xFFFF)))
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
        import time

        now = time.time()
        sec = int(now)
        usec = int((now - sec) * 1_000_000)
        try:
            os.write(
                self._fd, struct.pack(EVDEV_EVENT_FORMAT, sec, usec, EV_FF, code, value)
            )
        except OSError:
            return
