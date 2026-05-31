# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Interactive steering-wheel + pedal calibration CLI.

Walks the user through:

1. Picking a wheel device (auto-detect by movement when more than one
   joystick-like evdev device is present, with a numbered menu fallback).
2. Calibrating the steering axis (centre / full-left / full-right).
3. Calibrating the throttle and brake pedals (released / pressed).
4. Optionally testing force feedback.
5. Writing a YAML profile to ``configs/wheels/<name>.yaml`` that the
   :mod:`omnidreams.interactive_drive.demo` loader recognises.

The configurator deliberately keeps its runtime dependencies light --
only stdlib + PyYAML -- so it can be invoked on hosts that don't have
the ``interactive-drive`` (slangpy) extra installed; the goal is for
users to be able to calibrate their wheel even before the full demo
stack is ready.

Entry point: ``interactive-drive-configure-wheel``.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import select
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from omnidreams.interactive_drive._evdev import (
    EV_ABS,
    EV_FF,
    EVDEV_EVENT_FORMAT,
    EVDEV_EVENT_SIZE,
    EVIOCSFF,
    FF_AUTOCENTER,
    FF_CONSTANT,
    FF_GAIN,
    AxisInfo,
    EvdevDevice,
    query_axis,
    scan_evdev_devices,
)

# Common analogue ABS axis codes we'll probe. Covers all the standard
# wheel + pedal layouts (ABS_X..ABS_BRAKE); deliberately excludes
# discrete hat axes (ABS_HAT0X = 0x10+).
_CANDIDATE_AXES: tuple[int, ...] = tuple(range(0x00, 0x0B))

_ABS_AXIS_NAMES: dict[int, str] = {
    0x00: "ABS_X",
    0x01: "ABS_Y",
    0x02: "ABS_Z",
    0x03: "ABS_RX",
    0x04: "ABS_RY",
    0x05: "ABS_RZ",
    0x06: "ABS_THROTTLE",
    0x07: "ABS_RUDDER",
    0x08: "ABS_WHEEL",
    0x09: "ABS_GAS",
    0x0A: "ABS_BRAKE",
}


@dataclass(frozen=True)
class _AxisSnapshot:
    """Per-axis raw value sampled at a calibration checkpoint."""

    values: dict[int, int]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("\n[configure-wheel] aborted.", file=sys.stderr)
        return 130


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interactive-drive-configure-wheel",
        description=(
            "Interactive steering-wheel + pedal calibration. Writes a YAML "
            "profile compatible with `interactive-drive --wheel-profile`."
        ),
    )
    parser.add_argument(
        "--device",
        type=Path,
        default=None,
        help=(
            "Skip device selection and use this evdev path directly "
            "(e.g. /dev/input/event5 or /dev/input/by-id/usb-...-event-joystick)."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Profile name (and YAML filename stem). Defaults to a slug of "
            "the device's evdev name."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write the YAML into. Defaults to the bundled "
            "`configs/wheels/` directory inside the installed package so "
            "the demo's `--wheel-profile auto` picks it up automatically."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing profile YAML without prompting.",
    )
    parser.add_argument(
        "--no-default",
        action="store_true",
        help=(
            "Do not mark the generated profile as the default. Default "
            "profiles are matched first during `--wheel-profile auto` "
            "detection; useful when you're configuring a secondary wheel."
        ),
    )
    parser.add_argument(
        "--skip-ffb-test",
        action="store_true",
        help=(
            "Skip the force-feedback pulse test and write `ffb.enabled: false`. "
            "Useful on wheels without a motor or in CI."
        ),
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    devices = scan_evdev_devices()
    if not devices:
        print(
            "[configure-wheel] no evdev devices found under /dev/input/. "
            "Plug in your wheel and re-run.",
            file=sys.stderr,
        )
        return 1

    candidates = _filter_candidates(devices)
    if not candidates:
        print(
            "[configure-wheel] no devices with analogue ABS axes found. "
            "Wheels expose ABS_X/ABS_Z/ABS_RZ-style axes; if yours is plugged "
            "in but not listed, check `evtest` and udev permissions.",
            file=sys.stderr,
        )
        return 1

    device = _pick_device(args, candidates)
    if device is None:
        return 1

    print()
    print(f"Calibrating: {device.name}")
    print(f"Device path: {device.path}")

    axes = _detect_abs_axes(device.path)
    if len(axes) < 3:
        print(
            f"[configure-wheel] device only exposes {len(axes)} analogue axes "
            "(need >= 3 for steering + throttle + brake).",
            file=sys.stderr,
        )
        return 1

    try:
        steering_axis, invert_steering = _calibrate_steering(device.path, axes)
    except _CalibrationAborted as exc:
        print(f"\n[configure-wheel] {exc}", file=sys.stderr)
        return 1

    remaining = tuple(a for a in axes if a != steering_axis)
    try:
        throttle_axis, throttle_inverted = _calibrate_pedal(
            device.path, remaining, "throttle"
        )
    except _CalibrationAborted as exc:
        print(f"\n[configure-wheel] {exc}", file=sys.stderr)
        return 1

    remaining = tuple(a for a in remaining if a != throttle_axis)
    try:
        brake_axis, brake_inverted = _calibrate_pedal(device.path, remaining, "brake")
    except _CalibrationAborted as exc:
        print(f"\n[configure-wheel] {exc}", file=sys.stderr)
        return 1

    pedals_inverted = throttle_inverted
    if throttle_inverted != brake_inverted:
        print(
            "\n[configure-wheel] warning: throttle and brake pedal axes report "
            f"opposite inversion. Using throttle's reading (inverted={throttle_inverted}). "
            "If the brake feels wrong in the demo, edit the YAML to flip "
            "`pedal.inverted`."
        )

    if args.skip_ffb_test:
        ffb_enabled, ffb_mode, ffb_gain = False, "autocenter", 0.5
    else:
        ffb_enabled, ffb_mode, ffb_gain = _calibrate_ffb(device.path)

    profile_name = args.name or _slugify(device.name) or "custom-wheel"
    profile_name = _prompt_profile_name(profile_name)

    detection_patterns = _derive_detection_patterns(device.name)

    output_dir = args.output_dir or _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{profile_name}.yaml"
    if output_path.exists() and not args.force:
        ans = input(f"\n{output_path} already exists. Overwrite? [y/N]: ")
        if ans.strip().lower() != "y":
            print("[configure-wheel] aborted; existing profile left in place.")
            return 1

    _write_profile_yaml(
        output_path,
        name=profile_name,
        display_name=device.name,
        is_default=not args.no_default,
        detection_patterns=detection_patterns,
        steering_axis=steering_axis,
        throttle_axis=throttle_axis,
        brake_axis=brake_axis,
        pedals_inverted=pedals_inverted,
        invert_steering=invert_steering,
        ffb_enabled=ffb_enabled,
        ffb_mode=ffb_mode,
        ffb_gain=ffb_gain,
    )

    print(f"\nWheel profile written to {output_path}")
    print(
        "You can now launch the demo with auto-detection:\n"
        "    uv run --package flashdreams-omnidreams interactive-drive\n"
        f"or pin the profile explicitly:\n"
        f"    uv run --package flashdreams-omnidreams interactive-drive --wheel-profile {profile_name}"
    )
    return 0


# ---------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------


def _filter_candidates(devices: tuple[EvdevDevice, ...]) -> tuple[EvdevDevice, ...]:
    """Drop devices that don't expose any analogue ABS axes.

    Most evdev nodes on a typical desktop are keyboards / mice / power
    buttons; filtering them out keeps the menu short and makes the
    movement-based auto-detect unambiguous.
    """
    return tuple(d for d in devices if _detect_abs_axes(d.path))


def _detect_abs_axes(path: Path) -> tuple[int, ...]:
    return tuple(a for a in _CANDIDATE_AXES if query_axis(path, a) is not None)


def _pick_device(
    args: argparse.Namespace, candidates: tuple[EvdevDevice, ...]
) -> EvdevDevice | None:
    if args.device is not None:
        for device in candidates:
            if device.path == args.device:
                return device
            try:
                if device.path.resolve() == args.device.resolve():
                    return device
            except OSError:
                continue
        print(
            f"[configure-wheel] --device {args.device} did not match any "
            "joystick-like evdev node. Available candidates:",
            file=sys.stderr,
        )
        for device in candidates:
            print(f"  {device.path}  ({device.name})", file=sys.stderr)
        return None

    if len(candidates) == 1:
        only = candidates[0]
        print(f"Found one candidate: {only.name} ({only.path})")
        if _confirm("Use this device?", default=True):
            return only
        return None

    print("\nMultiple candidate devices found:")
    for i, device in enumerate(candidates, start=1):
        axes = _detect_abs_axes(device.path)
        axis_str = ", ".join(_ABS_AXIS_NAMES.get(a, f"0x{a:02x}") for a in axes)
        print(f"  [{i}] {device.name}")
        print(f"       path: {device.path}")
        print(f"       axes: {axis_str}")

    auto = _auto_detect_by_movement(candidates)
    if auto is not None:
        print(f"\nDetected movement on: {auto.name} ({auto.path})")
        if _confirm("Use this device?", default=True):
            return auto
    return _pick_from_menu(candidates)


def _auto_detect_by_movement(
    candidates: tuple[EvdevDevice, ...], duration_s: float = 4.0
) -> EvdevDevice | None:
    """Return the candidate device with the largest ABS-event traffic.

    Opens every candidate device, polls all fds for ``duration_s``
    seconds while prompting the user to wiggle their wheel, and ranks
    devices by cumulative absolute axis delta. Falls back to ``None``
    when no device shows meaningful movement (caller then offers a
    numbered menu).
    """
    print(
        f"\nGently turn the steering wheel left and right for the next "
        f"{int(duration_s)} seconds so I can identify your device..."
    )
    fds: dict[int, EvdevDevice] = {}
    try:
        for device in candidates:
            try:
                fd = os.open(device.path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError:
                continue
            fds[fd] = device
        if not fds:
            return None

        movement: dict[Path, int] = {device.path: 0 for device in candidates}
        last_values: dict[tuple[Path, int], int] = {}
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            timeout = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select(list(fds), [], [], min(0.2, timeout))
            for fd in readable:
                device = fds[fd]
                try:
                    data = os.read(fd, EVDEV_EVENT_SIZE * 32)
                except (BlockingIOError, OSError):
                    continue
                for off in range(0, len(data) - EVDEV_EVENT_SIZE + 1, EVDEV_EVENT_SIZE):
                    _, _, event_type, code, value = struct.unpack(
                        EVDEV_EVENT_FORMAT, data[off : off + EVDEV_EVENT_SIZE]
                    )
                    if event_type != EV_ABS:
                        continue
                    key = (device.path, int(code))
                    prev = last_values.get(key)
                    if prev is not None:
                        movement[device.path] += abs(int(value) - prev)
                    last_values[key] = int(value)

        ranked = sorted(movement.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked or ranked[0][1] < 200:
            return None
        # Require the winner to be clearly above the runner-up to avoid
        # mistaking a noisy axis on the wrong device for real input.
        if len(ranked) > 1 and ranked[0][1] < ranked[1][1] * 3:
            return None
        winning_path = ranked[0][0]
        for device in candidates:
            if device.path == winning_path:
                return device
        return None
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def _pick_from_menu(candidates: tuple[EvdevDevice, ...]) -> EvdevDevice | None:
    while True:
        try:
            choice = input(f"Pick a device [1-{len(candidates)}, q to quit]: ").strip()
        except EOFError:
            return None
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        try:
            idx = int(choice) - 1
        except ValueError:
            print("Please enter a number.")
            continue
        if 0 <= idx < len(candidates):
            return candidates[idx]
        print(f"Out of range. Pick 1-{len(candidates)}.")


# ---------------------------------------------------------------------
# Calibration steps
# ---------------------------------------------------------------------


class _CalibrationAborted(RuntimeError):
    pass


def _calibrate_steering(device_path: Path, axes: tuple[int, ...]) -> tuple[int, bool]:
    print("\nStep 1/3: Steering calibration")
    print(
        "  We only need full-left and full-right; you don't have to centre "
        "the wheel first (FFB wheels rarely sit at their true centre at "
        "rest) and you don't have to hit the exact mechanical extreme -- "
        "the runtime queries the device's reported axis range via "
        "EVIOCGABS to scale normalised steering."
    )

    print("\n  1a. Turn the wheel LEFT and HOLD.")
    _wait_for_enter("      Press Enter while holding the wheel left")
    left = _snapshot_axes(device_path, axes)

    print("\n  1b. Turn the wheel RIGHT and HOLD.")
    _wait_for_enter("      Press Enter while holding the wheel right")
    right = _snapshot_axes(device_path, axes)

    deltas = {a: abs(left.values[a] - right.values[a]) for a in axes}
    steering_axis, max_delta = max(deltas.items(), key=lambda kv: kv[1])
    if max_delta < 200:
        raise _CalibrationAborted(
            "no axis moved meaningfully between left and right. "
            "Make sure the wheel is plugged in and try again."
        )

    # Demo convention (see _normalize_steering + KeyboardDriveState):
    # ``steer > 0`` means "user turning left", and ``_normalize_steering``
    # produces ``(raw - axis_centre) / (axis_span / 2)`` then flips when
    # ``invert_steering`` is set. So when the user turns LEFT:
    #   raw_left > raw_right -> raw naturally exceeds centre on left  -> no flip
    #   raw_left < raw_right -> raw is below centre on left          -> flip
    invert_steering = left.values[steering_axis] < right.values[steering_axis]

    axis_name = _ABS_AXIS_NAMES.get(steering_axis, f"0x{steering_axis:02x}")
    print(
        f"\n  -> steering axis = 0x{steering_axis:02x} ({axis_name}), "
        f"invert_steering = {str(invert_steering).lower()}"
    )
    return steering_axis, invert_steering


def _calibrate_pedal(
    device_path: Path, candidates: tuple[int, ...], label: str
) -> tuple[int, bool]:
    step_num = {"throttle": "2/3", "brake": "3/3"}.get(label, "?/?")
    print(f"\nStep {step_num}: {label.capitalize()} pedal calibration")

    print(f"\n  Fully RELEASE the {label} pedal.")
    _wait_for_enter(f"      Press Enter while the {label} pedal is released")
    released = _snapshot_axes(device_path, candidates)

    print(f"\n  Press the {label} pedal all the way DOWN and HOLD.")
    _wait_for_enter(f"      Press Enter while holding the {label} pedal down")
    pressed = _snapshot_axes(device_path, candidates)

    deltas = {a: abs(released.values[a] - pressed.values[a]) for a in candidates}
    pedal_axis, max_delta = max(deltas.items(), key=lambda kv: kv[1])
    if max_delta < 100:
        raise _CalibrationAborted(
            f"no axis moved meaningfully when pressing the {label} pedal. "
            "Make sure your foot fully releases and fully presses the pedal."
        )

    # ``inverted_pedals=True`` means released=max, pressed=min (Thrustmaster
    # convention). Detect by comparing the two raw readings.
    inverted = released.values[pedal_axis] > pressed.values[pedal_axis]
    axis_name = _ABS_AXIS_NAMES.get(pedal_axis, f"0x{pedal_axis:02x}")
    print(
        f"\n  -> {label} axis = 0x{pedal_axis:02x} ({axis_name}), "
        f"pedal_inverted = {str(inverted).lower()}"
    )
    return pedal_axis, inverted


def _calibrate_ffb(device_path: Path) -> tuple[bool, str, float]:
    """Probe both FFB strategies and return ``(enabled, mode, gain)``.

    Order matters: we try ``FF_CONSTANT`` first because it works on
    *every* wheel base we support (Fanatec, Simagic, Moza, Thrustmaster,
    Logitech). ``FF_AUTOCENTER`` is the fallback for drivers that
    accept the in-kernel autocenter effect but do not implement
    ``EVIOCSFF`` (older Logitech profiles, the in-kernel ``hid-tmff``).
    Fanatec's ``hid-fanatecff`` does the *opposite*: ``FF_AUTOCENTER``
    is silently accepted with no force produced, so without the
    constant-force probe Fanatec users would always get
    ``ffb.enabled: false`` here.
    """
    print("\nStep 4 (optional): Force feedback")
    if not _confirm(
        "  Run a brief FFB test? Keep hands clear of the wheel.", default=True
    ):
        return False, "autocenter", 0.5

    print(
        "\n  Trying constant-force effect first "
        "(works on Fanatec / Simagic / Moza and modern Thrustmaster)..."
    )
    constant_status = _pulse_constant_force(device_path)
    if constant_status == "ok":
        if _confirm("  Did the wheel push left and then right?", default=True):
            gain = _prompt_gain(
                "  Constant-force gain (0.0-3.0, default 1.0): ",
                default=1.0,
                lo=0.0,
                hi=3.0,
            )
            return True, "constant_force", gain
        print("  No constant-force response felt; trying autocenter mode.")
    elif constant_status == "permission":
        # Permission errors are global to the device, so the autocenter
        # probe would just hit the same wall. Bail out early with a
        # useful message.
        return False, "autocenter", 0.5
    else:
        print(
            "  Driver did not accept the constant-force effect upload; "
            "trying autocenter mode."
        )

    print(
        "\n  Trying autocenter effect "
        "(works on Thrustmaster / Logitech via the in-kernel autocenter handler)..."
    )
    autocenter_status = _pulse_autocenter(device_path)
    if autocenter_status == "ok":
        if _confirm("  Did the wheel try to recentre itself?", default=True):
            gain = _prompt_gain(
                "  Autocenter gain (0.0-1.0, default 0.6): ",
                default=0.6,
                lo=0.0,
                hi=1.0,
            )
            return True, "autocenter", gain
        print("  No autocenter response either.")

    print("  Setting ffb.enabled: false; you can hand-edit the YAML later.")
    return False, "autocenter", 0.5


def _pulse_constant_force(device_path: Path) -> str:
    """Upload a ``FF_CONSTANT`` effect, pulse it left then right.

    Returns ``"ok"`` if the kernel accepted the effect upload (whether
    or not the user actually felt the wheel move), ``"permission"`` if
    we couldn't even open the device, or ``"unsupported"`` if the
    driver rejected ``EVIOCSFF``.
    """
    try:
        fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    except PermissionError:
        print(
            "  Permission denied opening the device for FFB. Add yourself to "
            "the `input` group or adjust udev rules, then re-run."
        )
        return "permission"
    except OSError as exc:
        print(f"  Could not open device for FFB ({exc}).")
        return "unsupported"

    try:
        effect_id = _upload_ff_constant(fd, effect_id=-1, level=1)
        if effect_id < 0:
            return "unsupported"
        # Play the effect once; subsequent EVIOCSFF re-uploads update
        # the level without needing another play event.
        _send_ev_ff_event(fd, code=effect_id, value=1)
        print("    pushing left for ~1 s...")
        _upload_ff_constant(fd, effect_id=effect_id, level=0x7FFF)
        time.sleep(1.0)
        print("    pushing right for ~1 s...")
        _upload_ff_constant(fd, effect_id=effect_id, level=-0x7FFF)
        time.sleep(1.0)
        _upload_ff_constant(fd, effect_id=effect_id, level=0)
        _send_ev_ff_event(fd, code=effect_id, value=0)
        return "ok"
    except OSError as exc:
        print(f"  FFB write failed ({exc}).")
        return "unsupported"
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _pulse_autocenter(device_path: Path) -> str:
    """Pulse ``FF_AUTOCENTER`` at max strength for ~2 s.

    Returns ``"ok"`` / ``"permission"`` / ``"unsupported"`` with the
    same semantics as :func:`_pulse_constant_force`.
    """
    try:
        fd = os.open(device_path, os.O_RDWR | os.O_NONBLOCK)
    except PermissionError:
        print(
            "  Permission denied opening the device for FFB. Add yourself to "
            "the `input` group or adjust udev rules, then re-run."
        )
        return "permission"
    except OSError as exc:
        print(f"  Could not open device for FFB ({exc}).")
        return "unsupported"

    try:
        _send_ev_ff_event(fd, code=FF_GAIN, value=0xFFFF)
        # Re-send the same strength every ~0.5 s in case the device's
        # FFB state machine expires stale autocenter values (some
        # kernels reset autocenter after ~1 s of no writes).
        for _ in range(4):
            _send_ev_ff_event(fd, code=FF_AUTOCENTER, value=0xFFFF)
            time.sleep(0.5)
        _send_ev_ff_event(fd, code=FF_AUTOCENTER, value=0)
        return "ok"
    except OSError as exc:
        print(f"  FFB write failed ({exc}).")
        return "unsupported"
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


# ``struct ff_effect`` size on 64-bit Linux; matches the kernel's
# definition referenced by ``EVIOCSFF``. ``constant.level`` lives at
# offset 16 (after the 2-byte alignment pad following replay.delay).
_FF_EFFECT_STRUCT_SIZE = 48
_FF_DIRECTION_EAST = 0x4000


def _upload_ff_constant(fd: int, *, effect_id: int, level: int) -> int:
    """``EVIOCSFF`` an ``ff_effect`` with the given constant level.

    Pass ``effect_id=-1`` on the first call to let the kernel assign
    an id; reuse the returned id for subsequent level updates so the
    driver doesn't allocate a fresh effect every tick. Returns the
    assigned id, or -1 if the upload was rejected.
    """
    buf = bytearray(_FF_EFFECT_STRUCT_SIZE)
    struct.pack_into("Hh", buf, 0, FF_CONSTANT, effect_id)
    struct.pack_into("H", buf, 4, _FF_DIRECTION_EAST)
    struct.pack_into("h", buf, 16, max(-0x7FFF, min(0x7FFF, level)))
    try:
        fcntl.ioctl(fd, EVIOCSFF, buf)
    except OSError:
        return -1
    return struct.unpack_from("Hh", buf, 0)[1]


def _send_ev_ff_event(fd: int, *, code: int, value: int) -> None:
    now = time.time()
    sec = int(now)
    usec = int((now - sec) * 1_000_000)
    os.write(
        fd,
        struct.pack(EVDEV_EVENT_FORMAT, sec, usec, EV_FF, code, value),
    )


def _prompt_gain(prompt: str, *, default: float, lo: float, hi: float) -> float:
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            return default
        if not raw:
            return default
        try:
            gain = float(raw)
        except ValueError:
            print(f"    Please enter a number between {lo} and {hi}.")
            continue
        if lo <= gain <= hi:
            return gain
        print(f"    Out of range; pick a value between {lo} and {hi}.")


# ---------------------------------------------------------------------
# Snapshot + raw-axis sampling
# ---------------------------------------------------------------------


def _snapshot_axes(device_path: Path, axes: tuple[int, ...]) -> _AxisSnapshot:
    """Read the current raw value for every requested ABS axis.

    Uses ``EVIOCGABS`` (current value field) so the snapshot reflects
    the live state without needing to consume the event stream. The
    kernel updates the ``input_absinfo.value`` field on every absolute
    axis event, so this is equivalent to "read the latest event" for
    all practical purposes.
    """
    values: dict[int, int] = {}
    for axis in axes:
        info: AxisInfo | None = query_axis(device_path, axis)
        if info is None:
            continue
        values[axis] = info.value
    return _AxisSnapshot(values=values)


# ---------------------------------------------------------------------
# Profile naming + YAML emission
# ---------------------------------------------------------------------


def _prompt_profile_name(default: str) -> str:
    while True:
        try:
            raw = input(f"\nProfile name [{default}]: ").strip()
        except EOFError:
            return default
        candidate = raw or default
        if re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
            return candidate
        print(
            "  Profile name must contain only letters, digits, underscore and "
            "hyphen (it becomes a YAML filename)."
        )


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return slug[:48]


def _derive_detection_patterns(name: str) -> tuple[str, ...]:
    """Produce a small ranked list of substring patterns from a device name.

    Matching in ``_device_matches_profile`` is case-insensitive substring;
    ranking from most-specific (exact full name) to most-generic
    (manufacturer alone) maximises the chance that a future kernel /
    udev rename still triggers auto-detect.
    """
    patterns: list[str] = [name]
    words = name.split()
    if len(words) >= 3:
        patterns.append(" ".join(words[1:]))
    if len(words) >= 1:
        patterns.append(words[0])
    seen: set[str] = set()
    deduped: list[str] = []
    for pattern in patterns:
        if pattern and pattern not in seen:
            seen.add(pattern)
            deduped.append(pattern)
    return tuple(deduped)


def _default_output_dir() -> Path:
    # Resolved relative to *this* module so the YAML lands next to the
    # rest of the bundled package data (``configs/wheels/``). The demo's
    # ``--wheel-profiles-dir`` default points at the same location.
    return Path(__file__).resolve().parent / "configs" / "wheels"


def _write_profile_yaml(
    path: Path,
    *,
    name: str,
    display_name: str,
    is_default: bool,
    detection_patterns: tuple[str, ...],
    steering_axis: int,
    throttle_axis: int,
    brake_axis: int,
    pedals_inverted: bool,
    invert_steering: bool,
    ffb_enabled: bool,
    ffb_mode: str,
    ffb_gain: float,
    threshold: float = 0.12,
) -> None:
    """Emit YAML in the same hand-written format the loader prefers.

    PyYAML's ``safe_dump`` works fine for the loader, but it strips the
    hex literals and inline ABS-name comments we want for hand-editing.
    Writing the file by hand keeps the layout consistent with the
    existing in-tree profiles (see the original ``thrustmaster.yaml``
    in the legacy ``omni-dreams`` sample).
    """
    axis_lines = [
        f"  steering: 0x{steering_axis:02x}   # {_ABS_AXIS_NAMES.get(steering_axis, '?')}",
        f"  throttle: 0x{throttle_axis:02x}   # {_ABS_AXIS_NAMES.get(throttle_axis, '?')}",
        f"  brake:    0x{brake_axis:02x}   # {_ABS_AXIS_NAMES.get(brake_axis, '?')}",
    ]
    pedal_comment = (
        "released=max, pressed=min" if pedals_inverted else "released=min, pressed=max"
    )

    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "#",
        "# Generated by `interactive-drive-configure-wheel`. Re-run that script",
        "# any time the device's evdev axis mapping changes, or hand-edit the",
        "# values below.",
        "",
        f"name: {name}",
        f'display_name: "{display_name}"',
        f"is_default: {str(is_default).lower()}",
        "",
        "detection_patterns:",
        *(f'  - "{_yaml_quote(pattern)}"' for pattern in detection_patterns),
        "",
        "axis_map:",
        *axis_lines,
        "",
        "pedal:",
        f"  inverted: {str(pedals_inverted).lower()}   # {pedal_comment}",
        "",
        f"invert_steering: {str(invert_steering).lower()}",
        "",
        "ffb:",
        f"  mode: {ffb_mode}   # {_FFB_MODE_COMMENTS.get(ffb_mode, '')}".rstrip(),
        f"  enabled: {str(ffb_enabled).lower()}",
        f"  gain: {_format_float(ffb_gain)}",
        "",
        f"threshold: {_format_float(threshold)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


_FFB_MODE_COMMENTS: dict[str, str] = {
    "autocenter": "in-kernel FF_AUTOCENTER (Thrustmaster, Logitech)",
    "constant_force": "EVIOCSFF FF_CONSTANT effect (Fanatec, Simagic, Moza, "
    "Thrustmaster)",
}


def _yaml_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


# ---------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------


def _wait_for_enter(prompt: str) -> None:
    try:
        input(prompt + ": ")
    except EOFError:
        raise _CalibrationAborted("stdin closed before calibration completed.")


def _confirm(prompt: str, *, default: bool) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        ans = input(prompt + suffix + ": ").strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in {"y", "yes"}


if __name__ == "__main__":
    sys.exit(main())
