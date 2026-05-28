# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import array
import fcntl
import io
import math
import os
import select
import struct
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omnidreams import scenes as _scenes
from omnidreams.interactive_drive import cli as _cli
from omnidreams.interactive_drive.app import InteractiveDriveApp
from omnidreams.interactive_drive.config import BevConfig, RasterConfig
from omnidreams.scenes import normalise_scene_uuid, scenes_cache_root
from PIL import Image

EVDEV_EVENT_FORMAT = "llHHi"
EVDEV_EVENT_SIZE = struct.calcsize(EVDEV_EVENT_FORMAT)
EV_ABS = 0x03
EV_FF = 0x15
FF_AUTOCENTER = 0x61
FF_GAIN = 0x60
EVIOCGABS = lambda axis: 0x80184540 + axis  # noqa: E731

# Width of the right-side HUD panel that holds the steering wheel,
# pedals, speed digit and BEV minimap. The camera area fills the rest
# of the live screen width. Pinned at 500 px because the panel content
# (wheel asset, pedal pngs) is asset-driven and doesn't reflow.
HUD_PANEL_WIDTH = 500
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_THUMB_SIZE = (140, 64)
KEYBOARD_STEER_SCALE = 0.75
KEYBOARD_STEER_RATE_PER_S = 0.6
KEYBOARD_STEER_RETURN_RATE_PER_S = 1.4
# BEV minimap panel sits at the bottom of the right HUD column.
# Geometry is hand-tuned to leave ~12px gaps to the pedals/edges and
# keeps roughly square aspect to match the BEV camera output.
BEV_PANEL_TOP_GAP = 12
BEV_PANEL_SIDE_MARGIN = 14
BEV_PANEL_BOTTOM_MARGIN = 12
BEV_PANEL_MIN_HEIGHT = 100

# Google-Maps "land" colour: warm cream, slightly desaturated. Matches the
# off-white background on Google Maps' default day-mode tiles. Black /
# unrendered regions of the BEV image get blended toward this colour by
# :func:`_apply_googlemaps_filter`.
GMAPS_LAND_RGB = (234, 226, 209)
# Highlight tint for road paint / lane markings. Google Maps draws minor
# roads in pale grey; we keep the rasterizer's whites/yellows but blend
# them slightly toward this so they don't feel neon-bright on the cream.
GMAPS_ROAD_RGB = (252, 250, 244)
# Substitute colour for magenta-rendered road boundaries. Soft warm grey
# slightly darker than the cream land so the boundary still reads as a
# road edge but with low enough contrast that aliasing on diagonals is
# imperceptible. The cream-vs-magenta jump (~150 lightness) was the
# dominant aliasing offender; cream-vs-grey is ~30, well below the
# threshold most viewers can resolve at panel size.
GMAPS_BOUNDARY_GREY_RGB = (170, 165, 155)
# Pre-built float32 arrays used by ``_apply_googlemaps_filter`` so the
# numpy expression that runs once per BEV frame doesn't re-allocate
# these constant 3-vectors each call.
_GMAPS_LAND_FLOAT = np.array(GMAPS_LAND_RGB, dtype=np.float32)
_GMAPS_BOUNDARY_GREY_FLOAT = np.array(GMAPS_BOUNDARY_GREY_RGB, dtype=np.float32)
_GMAPS_TINTED_MUL = (
    0.55 + 0.45 * np.array(GMAPS_ROAD_RGB, dtype=np.float32) / 255.0
).astype(np.float32)

# Pull BEV camera defaults from the canonical :class:`BevConfig` so the
# HUD's ego-marker placement automatically follows changes to the rasterizer
# default. The marker sits at the rig's image projection: pure top-down
# (tilt = 0) places it in the centre; positive tilt pushes it lower in the
# frame because the camera now sees more ahead of the rig.
_BEV_DEFAULTS = BevConfig()
BEV_FOV_DEG = _BEV_DEFAULTS.fov_deg
BEV_TILT_DEG = _BEV_DEFAULTS.tilt_deg


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


@dataclass(frozen=True)
class SceneOption:
    label: str
    path: Path
    variants: tuple[str, ...]
    thumbnail: Image.Image | None = None


@dataclass
class WheelState:
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    target_speed_mps: float = 0.0
    connected: bool = False


class KeyboardDriveState:
    def __init__(self, control: Any) -> None:
        # ``control`` quacks like the supervisor-era ``ControlClient``:
        # it has ``set_drive(steer, throttle, brake)``. In the slangpy
        # HUD path it's
        # :class:`~omnidreams.interactive_drive.slangpy_hud_presenter.KeyboardStateDriveSink`,
        # which writes straight into the in-process ``KeyboardState``.
        self._control = control
        self._pressed: set[str] = set()
        self._state = WheelState()
        self._last_update_s = time.monotonic()

    @property
    def state(self) -> WheelState:
        return WheelState(**self._state.__dict__)

    def set_key(self, keysym: str, down: bool) -> bool:
        key = _keyboard_drive_key(keysym)
        if key is None:
            return False
        if down:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)
        return True

    def update(self) -> WheelState:
        now = time.monotonic()
        dt = max(0.0, min(0.1, now - self._last_update_s))
        self._last_update_s = now

        target_steer = 0.0
        if {"a", "left"} & self._pressed:
            target_steer += KEYBOARD_STEER_SCALE
        if {"d", "right"} & self._pressed:
            target_steer -= KEYBOARD_STEER_SCALE
        rate = (
            KEYBOARD_STEER_RATE_PER_S
            if abs(target_steer) > 0
            else KEYBOARD_STEER_RETURN_RATE_PER_S
        )
        steer = _move_towards(self._state.steering, target_steer, rate * dt)
        throttle = 1.0 if {"w", "up"} & self._pressed else 0.0
        brake = 1.0 if {"s", "down", "space"} & self._pressed else 0.0
        target_speed = self._update_target_speed(throttle=throttle, brake=brake, dt=dt)
        self._state = WheelState(
            steering=steer,
            throttle=throttle,
            brake=brake,
            target_speed_mps=target_speed,
            connected=False,
        )
        self._control.set_drive(steer=steer, throttle=throttle, brake=brake)
        return self.state

    def clear(self) -> None:
        self._pressed.clear()
        self._state = WheelState()
        self._control.set_drive(steer=0.0, throttle=0.0, brake=0.0)

    def _update_target_speed(
        self, *, throttle: float, brake: float, dt: float
    ) -> float:
        speed = self._state.target_speed_mps
        if throttle > 0.01 and brake <= 0.05:
            accel = 2.0 * throttle * dt
            current = abs(speed)
            high_speed_knee = 22.35
            if current < high_speed_knee:
                taper = max(0.2, 1.0 - (current / high_speed_knee) ** 2 * 0.5)
            else:
                excess = (current - high_speed_knee) / max(1e-6, 36.0 - high_speed_knee)
                taper = max(0.05, 0.5 * (1.0 - excess) ** 3)
            speed += accel * taper
        elif brake > 0.01:
            speed = max(0.0, speed - 12.0 * brake * dt)
        else:
            creep_target = 4.47
            if speed < creep_target + 0.1:
                speed += (creep_target - speed) * 0.18 * dt
            else:
                speed = max(0.0, speed - 0.5 * dt)
        return max(0.0, min(36.0, speed))


@dataclass(frozen=True)
class ControlAssets:
    steering_wheel: Image.Image | None
    throttle_pressed: Image.Image | None
    throttle_unpressed: Image.Image | None
    brake_pressed: Image.Image | None
    brake_unpressed: Image.Image | None

    @property
    def complete(self) -> bool:
        return (
            self.steering_wheel is not None
            and self.throttle_pressed is not None
            and self.throttle_unpressed is not None
            and self.brake_pressed is not None
            and self.brake_unpressed is not None
        )


class WheelBridge:
    def __init__(
        self,
        *,
        device_path: Path,
        profile: WheelProfile,
        control: Any,
    ) -> None:
        # ``control`` quacks like the supervisor-era ``ControlClient``:
        # it has ``set_drive(steer, throttle, brake)`` and
        # ``release_all()``. In-process the slangpy HUD passes a
        # :class:`KeyboardStateDriveSink`; the wheel reader thread
        # then writes to ``KeyboardState`` directly with no HTTP hop.
        self._device_path = device_path
        self._profile = profile
        self._control = control
        self._steering_axis = int(profile.axis_map["steering"])
        self._throttle_axis = int(profile.axis_map["throttle"])
        self._brake_axis = int(profile.axis_map["brake"])
        self._inverted_pedals = bool(profile.inverted_pedals)
        self._invert_steering = bool(profile.invert_steering)
        self._threshold = float(profile.threshold)
        self._ffb = AutocenterFFB()
        self._axis_ranges: dict[int, AxisRange] = {}
        self._raw_axes: dict[int, int] = {}
        self._state = WheelState()
        self._state_lock = threading.Lock()
        self._last_update_s = time.monotonic()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> WheelState:
        with self._state_lock:
            return WheelState(**self._state.__dict__)

    def start(self) -> None:
        self._axis_ranges = {
            axis: _query_axis_range(self._device_path, axis)
            or AxisRange(minimum=0, maximum=65535)
            for axis in (self._steering_axis, self._throttle_axis, self._brake_axis)
        }
        self._raw_axes = {
            self._steering_axis: int(self._axis_ranges[self._steering_axis].center),
            self._throttle_axis: self._released_pedal_raw(self._throttle_axis),
            self._brake_axis: self._released_pedal_raw(self._brake_axis),
        }
        if self._profile.ffb_enabled:
            self._ffb.init(self._device_path, self._profile.ffb_gain)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="interactive-drive-wheel", daemon=True
        )
        self._thread.start()
        print(
            f"[demo] wheel profile={self._profile.name} device={self._device_path} "
            f"axes={self._profile.axis_map} ranges={self._axis_ranges}",
            flush=True,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._ffb.cleanup()
        self._control.release_all()

    def _run(self) -> None:
        try:
            fd = os.open(self._device_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            print(
                f"[demo] failed to open wheel device {self._device_path}: {exc}",
                flush=True,
            )
            return
        try:
            with self._state_lock:
                self._state.connected = True
            while not self._stop_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.02)
                if readable:
                    self._read_events(fd)
                self._publish_controls()
        finally:
            os.close(fd)
            with self._state_lock:
                self._state.connected = False

    def _read_events(self, fd: int) -> None:
        try:
            data = os.read(fd, EVDEV_EVENT_SIZE * 32)
        except BlockingIOError:
            return
        for offset in range(0, len(data) - EVDEV_EVENT_SIZE + 1, EVDEV_EVENT_SIZE):
            _, _, event_type, code, value = struct.unpack(
                EVDEV_EVENT_FORMAT, data[offset : offset + EVDEV_EVENT_SIZE]
            )
            if event_type == EV_ABS:
                self._raw_axes[int(code)] = int(value)

    def _publish_controls(self) -> None:
        steering = self._normalize_steering(self._raw_axes[self._steering_axis])
        throttle = self._normalize_pedal(
            self._throttle_axis, self._raw_axes[self._throttle_axis]
        )
        brake = self._normalize_pedal(
            self._brake_axis, self._raw_axes[self._brake_axis]
        )
        target_speed = self._update_target_speed(throttle=throttle, brake=brake)
        with self._state_lock:
            self._state.steering = steering
            self._state.throttle = throttle
            self._state.brake = brake
            self._state.target_speed_mps = target_speed

        self._control.set_drive(steer=steering, throttle=throttle, brake=brake)
        self._ffb.update(abs(target_speed), gain=self._profile.ffb_gain)

    def _normalize_steering(self, raw: int) -> float:
        axis_range = self._axis_ranges[self._steering_axis]
        value = (float(raw) - axis_range.center) / (axis_range.span * 0.5)
        if self._invert_steering:
            value = -value
        return max(-1.0, min(1.0, value))

    def _normalize_pedal(self, axis: int, raw: int) -> float:
        axis_range = self._axis_ranges[axis]
        if self._inverted_pedals:
            value = (float(axis_range.maximum) - float(raw)) / axis_range.span
        else:
            value = (float(raw) - float(axis_range.minimum)) / axis_range.span
        return max(0.0, min(1.0, value))

    def _released_pedal_raw(self, axis: int) -> int:
        axis_range = self._axis_ranges[axis]
        return axis_range.maximum if self._inverted_pedals else axis_range.minimum

    def _update_target_speed(self, *, throttle: float, brake: float) -> float:
        now = time.monotonic()
        dt = max(0.0, min(0.1, now - self._last_update_s))
        self._last_update_s = now
        with self._state_lock:
            speed = self._state.target_speed_mps
        if throttle > 0.01 and brake <= 0.05:
            accel = 2.0 * throttle * dt
            current = abs(speed)
            high_speed_knee = 22.35
            if current < high_speed_knee:
                taper = max(0.2, 1.0 - (current / high_speed_knee) ** 2 * 0.5)
            else:
                excess = (current - high_speed_knee) / max(1e-6, 36.0 - high_speed_knee)
                taper = max(0.05, 0.5 * (1.0 - excess) ** 3)
            speed += accel * taper
        elif brake > 0.01:
            speed = max(0.0, speed - 12.0 * brake * dt)
        else:
            creep_target = 4.47  # 10 mph, matching the AlpaSim manual-driver creep.
            if speed < creep_target + 0.1:
                # Demo crawl should be gentle: a first-order approach that
                # takes several seconds to reach 10 mph from a stop.
                speed += (creep_target - speed) * 0.18 * dt
            else:
                speed = max(0.0, speed - 0.5 * dt)
        return max(0.0, min(36.0, speed))


class AutocenterFFB:
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
                self._fd, struct.pack(EVDEV_EVENT_FORMAT, sec, usec, EV_FF, code, value)
            )
        except OSError:
            return


def build_parser() -> argparse.ArgumentParser:
    """Build the unified ``interactive-drive`` argument parser.

    The parser is the union of three groups:

    * Backend args (``--scene``, ``--backend``, ``--manifest``,
      ``--bev``, ...) inherited verbatim from
      :func:`omnidreams.interactive_drive.cli.build_parser`. These
      flags apply whether the user runs the supervised HUD wrapper or
      the bare backend with ``--no-hud``.
    * Supervisor / HUD args (``--scene-dir``, ``--autoload-scene``,
      ``--cuda-visible-devices``, ``--wheel-*``, ``--no-wheel``) that
      only matter when a HUD viewer is running. They're harmlessly
      ignored under ``--no-hud``.
    * The ``--no-hud`` toggle itself, which falls through to the bare
      slangpy Vulkan window. Browser-stream use cases are served by
      ``omnidreams.webrtc.server`` instead.
    """
    parser = _cli.build_parser()
    # Demo-friendly defaults: most users want the world model and the
    # bundled example manifest. The bare cli still defaults to
    # ``raster`` / ``manifest=None`` for unit-test friendliness.
    # Manifest path is rooted at the sample's own packaged ``configs/`` so
    # the default lands on the bundled YAML regardless of the user's cwd
    # (flashdreams workspaces run from the repo root, not the sample dir).
    parser.set_defaults(
        backend="omnidreams",
        manifest=_cli._PACKAGE_ROOT / "configs/example_world_model.yaml",
    )
    parser.description = (
        "Interactive driving demo. Default mode opens a slangpy HUD with"
        " scene/variant selector, BEV minimap, and steering / pedal"
        " overlays, all rendered into a single Vulkan swapchain. Pass"
        " --no-hud to drop the chrome and just open the bare slangpy"
        " Vulkan window. For browser / remote streaming use the separate"
        " ``omnidreams.webrtc.server`` entry point."
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help=(
            "Skip the HUD chrome and run the backend with a bare slangpy"
            " Vulkan window (matching the legacy lightweight demo)."
        ),
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=scenes_cache_root(),
        help=(
            "Directory of USDZ scenes shown in the HUD scene selector. "
            "Defaults to ``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/``, "
            "the shared cache root used by both this demo and the "
            "``omnidreams.webrtc.server`` scene pipeline."
        ),
    )
    parser.add_argument(
        "--autoload-scene",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start loading --scene immediately. By default the HUD opens on Load Scene.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="auto",
        help=(
            "CUDA_VISIBLE_DEVICES for the backend. ``auto`` (default) leaves"
            " whatever the user already exported untouched; a literal value"
            " (e.g. ``0`` or ``1``) is passed through verbatim; empty string"
            " forces the env var unset. The HUD does not auto-pick a GPU --"
            " set CUDA_VISIBLE_DEVICES (or pass an explicit value) on"
            " multi-GPU hosts where the default-zero pick is wrong."
        ),
    )
    parser.add_argument("--wheel-profile", default="auto")
    parser.add_argument(
        "--wheel-profiles-dir", type=Path, default=_cli._PACKAGE_ROOT / "configs/wheels"
    )
    parser.add_argument(
        "--control-assets-dir",
        type=Path,
        default=None,
        help="Directory containing AlpaSim wheel/pedal PNGs. Defaults to data/wheel_and_pedals if present.",
    )
    parser.add_argument(
        "--wheel-device",
        type=Path,
        default=None,
        help="Optional explicit evdev path. Auto-detect scans /dev/input/by-id first.",
    )
    parser.add_argument(
        "--wheel-steering-axis", type=_parse_axis, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--wheel-throttle-axis", type=_parse_axis, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--wheel-brake-axis", type=_parse_axis, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--wheel-pedals-inverted",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-wheel", action="store_true")
    return parser


def _maybe_autostage_scene(scene: Path) -> Path:
    """Auto-download a known scene UUID on first launch.

    Triggers only when ``scene`` lives under the shared scenes cache
    root (``$FLASHDREAMS_CACHE_DIR/omnidreams-scenes/``), is missing on
    disk, and the filename matches the ``clipgt-<uuid>.usdz`` convention
    used by ``nvidia/omni-dreams-scenes``. User-pointed external paths
    and non-clipgt filenames are returned unchanged so the demo's
    normal "file not found" error fires for them.

    The explicit ``omnidreams-prepare`` script remains the way to
    pre-stage arbitrary UUIDs and to pre-warm the Cosmos-Reason1 text
    encoder; this helper just covers the "I just ran ``interactive-drive``
    and the default scene isn't there yet" case so first-launch is a
    single command.
    """
    if scene.exists():
        return scene
    cache_dir = scenes_cache_root().resolve()
    if scene.resolve().parent != cache_dir:
        return scene
    stem = scene.stem
    if not stem.startswith("clipgt-"):
        return scene
    bare_uuid = normalise_scene_uuid(stem)
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            f"Scene '{scene.name}' is not staged yet and HF_TOKEN is not set.\n"
            "Either export HF_TOKEN to enable auto-staging on launch, or run:\n"
            f"  uv run --package flashdreams-omnidreams omnidreams-prepare --scene-uuid {bare_uuid}"
        )
    print(
        f"[interactive-drive] Scene '{stem}' not found locally; "
        "auto-staging from Hugging Face (one-time download, ~30 MB)..."
    )
    from omnidreams.prepare import stage_scene

    return stage_scene(bare_uuid, force=False)


def main() -> None:
    args = build_parser().parse_args()
    if not args.synthetic_scene:
        args.scene = _maybe_autostage_scene(args.scene)
    # ``--no-hud`` drops the slangpy HUD chrome and runs the backend
    # against a bare Vulkan window. Browser / remote use cases live
    # in the separate ``omnidreams.webrtc.server`` entry point.
    if args.no_hud:
        _cli.run(args)
        return

    _run_slangpy_hud(args)


def _run_slangpy_hud(args: argparse.Namespace) -> None:
    """Run the engine with the slangpy + PIL HUD presenter in one process.

    Replaces the supervised pygame-HUD architecture entirely. The engine
    runs on the main thread (matching ``--no-hud``'s topology, the only
    one we have empirical evidence for working on this hardware: pygame
    + Ludus + CUDA in one process consistently failed at the EGL or
    CUDA-GL interop layer).

    The function loops over scene-change requests so the user can pick
    a new scene from the HUD dropdown without the slangpy window
    closing and reopening. One ``SlangPyHudPresenter`` is constructed
    at startup and reused across many ``app.run()`` invocations -- one
    per scene the user picks. Each iteration tears down only the
    backend / pipeline / simulation, rebuilds them for the freshly
    selected scene, and hands them to a new
    :class:`InteractiveDriveApp` whose ``close_presenter_on_exit=False``
    keeps the presenter (and therefore the window) alive across the
    transition.

    The wheel is a long-lived resource too -- evdev fd, FFB context --
    so it's constructed once and rebound to each successive
    ``KeyboardState`` via the presenter's
    :meth:`SlangPyHudPresenter.bind_keyboard`. We rebuild the wheel
    bridge if the bind target differs because ``WheelBridge`` captures
    the sink at init.
    """
    from omnidreams.interactive_drive.input.keyboard import KeyboardState
    from omnidreams.interactive_drive.slangpy_hud_presenter import (
        KeyboardStateDriveSink,
        SlangPyHudPresenter,
    )

    _apply_cuda_visible_devices_inplace(args.cuda_visible_devices)
    _resolve_demo_paths(args)
    scene_options = _discover_scene_options(args.scene_dir, args.scene)
    if not args.scene.exists() and scene_options:
        args.scene = scene_options[0].path
    # Validate paths up front so a typo in ``--manifest`` /
    # ``--scene-dir`` / ``--control-assets-dir`` fails immediately,
    # before we open the slangpy window and the user wastes 30s on
    # world-model warmup that's about to ENOENT. Scene path is
    # validated lazily because ``_discover_scene_options`` already
    # backfills ``args.scene`` from the directory, so a missing
    # ``--scene`` is only fatal if the directory is empty too.
    if args.backend == "omnidreams":
        if args.manifest is None:
            raise SystemExit("--manifest is required for the omnidreams backend")
        if not args.manifest.exists():
            raise SystemExit(
                f"--manifest path does not exist: {args.manifest}"
                " (typo? expected something like configs/example_world_model.yaml)"
            )
    if not scene_options and not args.scene.exists():
        raise SystemExit(
            f"--scene path does not exist and --scene-dir contains no scenes: {args.scene}"
        )
    control_assets = _load_control_assets(args.control_assets_dir)
    wheel_selection = None if args.no_wheel else _select_wheel(args)

    # Construct the presenter UPFRONT, before any backend, so the demo
    # can open the HUD window in "Load Scene" mode and wait for the
    # user to pick a scene from the dropdown when ``--autoload-scene``
    # is off. The placeholder ``KeyboardState`` is rebound to each
    # successive ``InteractiveDriveApp``'s real keyboard via
    # ``presenter.bind_keyboard`` in the factory below; no engine is
    # listening to the placeholder, so events are harmlessly dropped
    # during the initial wait.
    placeholder_keyboard = KeyboardState()
    presenter = SlangPyHudPresenter(
        raster=RasterConfig(),
        keyboard=placeholder_keyboard,
        args=args,
        scene_options=scene_options,
        control_assets=control_assets,
        wheel=None,
    )
    wheel: Any = None

    def _factory(config: Any, keyboard: Any) -> Any:
        # Called once per ``InteractiveDriveApp.__init__``. The first
        # call lazily attaches the wheel (so the evdev reader thread
        # only starts running once the user has actually picked a
        # scene); subsequent calls rebind the wheel's drive sink to
        # the new keyboard without restarting the reader.
        nonlocal wheel
        if wheel is None and wheel_selection is not None:
            profile, device_path = wheel_selection
            wheel = WheelBridge(
                device_path=device_path,
                profile=profile,
                control=KeyboardStateDriveSink(keyboard),
            )
            wheel.start()
            presenter.set_wheel(wheel)
        elif wheel is not None:
            # Rebind the wheel's drive sink to the new keyboard. We
            # don't restart the wheel's evdev reader thread -- it's
            # state-machine-clean and the only thing tied to keyboard
            # is the sink it posts ``set_drive`` calls into.
            wheel._control = KeyboardStateDriveSink(keyboard)  # noqa: SLF001 -- see comment
        presenter.bind_keyboard(keyboard)
        return presenter

    try:
        # Initial scene-selection wait: if the user didn't pass
        # ``--autoload-scene``, open the HUD and let them pick a
        # scene from the dropdown. Skipping the autoload also lets
        # the user pick a different scene than ``--scene`` advertises
        # without re-running the binary.
        if not args.autoload_scene:
            request = presenter.wait_for_scene_selection()
            if request is None:
                return  # window closed before any scene was loaded
            scene_path, variant = request
            args.scene = scene_path
            args.variant = variant
            presenter.acknowledge_scene_change(scene_path, variant)

        while True:
            presenter.set_engine_active(True)
            config, backend = _cli.prepare_config_and_backend(args)
            app = InteractiveDriveApp(
                config=config,
                backend=backend,
                presenter_factory=_factory,
                close_presenter_on_exit=False,
            )
            app.run()
            presenter.set_engine_active(False)
            requested = presenter.pending_scene_change
            if requested is None:
                # User closed the window (X / ESC); we're done.
                break
            new_scene_path, new_variant = requested
            args.scene = new_scene_path
            args.variant = new_variant
            presenter.acknowledge_scene_change(new_scene_path, new_variant)
    finally:
        presenter.close()


def _apply_cuda_visible_devices_inplace(requested: str) -> None:
    """Resolve ``--cuda-visible-devices`` into the in-process ``os.environ``.

    In-process we mutate ``os.environ`` directly so torch / CUDA see the
    right device list before any backend construction. MUST run before
    ``_cli.run`` (which is what pulls in flashdreams /
    WorldModelRenderBackend / torch.cuda).

    ``auto`` is a no-op (leave whatever the user already exported alone).
    Earlier versions of this helper auto-picked GPU ``1`` on any
    multi-GPU host -- that assumed the RTX6000 + GB300 dev box layout
    and silently picked the wrong GPU on other multi-GPU machines.
    Users on hosts where the default-zero pick is wrong should export
    ``CUDA_VISIBLE_DEVICES`` themselves or pass an explicit value.
    """
    if requested == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return
    if requested != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = requested


def _resolve_demo_paths(args: argparse.Namespace) -> None:
    for attr in ("scene", "manifest", "scene_dir", "wheel_profiles_dir"):
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, _project_path(value))
    if args.control_assets_dir is not None:
        args.control_assets_dir = _project_path(args.control_assets_dir)


def _project_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _discover_scene_options(
    scene_dir: Path, selected_scene: Path
) -> tuple[SceneOption, ...]:
    paths: set[Path] = set()
    if selected_scene.exists():
        paths.add(selected_scene.resolve())
    if scene_dir.is_dir():
        paths.update(path.resolve() for path in scene_dir.glob("*.usdz"))
    if selected_scene.parent.is_dir():
        paths.update(path.resolve() for path in selected_scene.parent.glob("*.usdz"))
    options = tuple(
        SceneOption(
            label=_scene_label(path),
            path=path,
            variants=_discover_variants(path),
            thumbnail=_load_scene_thumbnail(path),
        )
        for path in sorted(paths)
    )
    print(
        "[demo] discovered scenes: "
        + (", ".join(scene.label for scene in options) if options else "<none>"),
        flush=True,
    )
    return options


def _scene_label(path: Path) -> str:
    scene_names = {
        "clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4": "Quiet Suburban Boulevard",
        "clipgt-7bd1eb2f-c375-44ee-b4ca-55473e0773a9": "Late Night Arrival in the Neighborhood",
        "clipgt-e2993759-36e1-4d97-868f-e2a737f1eb68": "Afternoon Commute Past the Park",
    }
    scene_id = path.stem
    return scene_names.get(scene_id, scene_id)


def _discover_variants(scene_path: Path) -> tuple[str, ...]:
    variants: set[str] = set()
    try:
        with zipfile.ZipFile(scene_path, "r") as zf:
            for name in zf.namelist():
                if "/" in name:
                    continue
                stem = Path(name).stem
                if name.startswith("first_image") and name.endswith(".png"):
                    variant = _scenes.variant_from_stem(stem, "first_image")
                elif name.startswith("prompt") and name.endswith(".txt"):
                    variant = _scenes.variant_from_stem(stem, "prompt")
                else:
                    continue
                if variant is not None:
                    variants.add(variant)
    except (OSError, zipfile.BadZipFile):
        return ("default",)
    if not variants:
        variants.add("default")
    if "default" not in variants:
        variants.add(sorted(variants)[0])
    return tuple(sorted(variants, key=lambda value: (value != "default", value)))


def _load_scene_thumbnail(scene_path: Path) -> Image.Image | None:
    try:
        with zipfile.ZipFile(scene_path, "r") as zf:
            names = [
                name
                for name in zf.namelist()
                if "/" not in name
                and name.startswith("first_image")
                and name.endswith(".png")
            ]
            if not names:
                return None
            name = "first_image.png" if "first_image.png" in names else sorted(names)[0]
            with Image.open(io.BytesIO(zf.read(name))) as image:
                return _make_thumbnail(image.convert("RGB"), SCENE_THUMB_SIZE)
    except (OSError, zipfile.BadZipFile):
        return None


def _make_thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = Image.new("RGB", size, (20, 20, 30))
    fitted = _fit_image(image, size)
    thumb.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return thumb


# ``_variant_from_stem`` removed in favour of the shared
# :func:`scene_loader.variant_from_stem` -- all three discovery paths
# (USDZ archives, unpacked scene dirs, HUD variant selector) now agree
# on the underscore-required convention. The previous fallback that
# accepted ``prompt1.txt`` as variant ``1`` was inconsistent with
# ``_discover_first_images`` and silently masked the
# ``prompt_<N>.txt`` -> ``_<N>`` bug on real scenes.


def _variant_label(variant: str) -> str:
    labels = {
        "default": "Default",
        "1": "Bright Midday Sun",
        "2": "Snowstorm",
        "3": "Night with Heavy Rain",
    }
    return labels.get(variant, variant)


def _select_wheel(args: argparse.Namespace) -> tuple[WheelProfile, Path] | None:
    profiles = _load_wheel_profiles(args.wheel_profiles_dir)
    profile = _profile_by_name(profiles, args.wheel_profile)
    device_path: Path | None = args.wheel_device

    if profile is None and device_path is not None:
        # ``--wheel-profile auto`` with an explicit ``--wheel-device``:
        # don't run the device-scan auto-detect (which would ignore the
        # user's path); just read the device name and match it against
        # the loaded profiles.
        profile = _profile_for_device(profiles, device_path)
        if profile is None:
            print(
                f"[demo] no wheel profile matched device {device_path}; "
                "pass --wheel-profile <name> explicitly",
                flush=True,
            )
            return None
    elif profile is None:
        selection = _detect_wheel(profiles)
        if selection is None:
            print(
                "[demo] no wheel detected; use --wheel-device or --no-wheel", flush=True
            )
            return None
        profile, device_path = selection
    elif device_path is None:
        device = _detect_device_for_profile(profile)
        if device is None:
            print(
                f"[demo] wheel profile {profile.name!r} did not match any evdev device",
                flush=True,
            )
            return None
        device_path = device.path

    assert device_path is not None
    profile = _apply_wheel_overrides(profile, args)
    return profile, device_path


def _profile_for_device(
    profiles: tuple[WheelProfile, ...], device_path: Path
) -> WheelProfile | None:
    """Pick the best profile for an explicit ``--wheel-device`` path.

    Prefers ``is_default``-flagged profiles (same priority order as
    :func:`_detect_wheel`) and matches by the device's reported evdev
    name. Returns ``None`` when no profile's detection patterns match.
    """
    name = _read_evdev_name(device_path)
    if name is None:
        return None
    fake_device = EvdevDevice(path=device_path, name=name)
    ordered = sorted(profiles, key=lambda p: p.is_default, reverse=True)
    for profile in ordered:
        if _device_matches_profile(fake_device, profile):
            return profile
    return None


def _load_wheel_profiles(profiles_dir: Path) -> tuple[WheelProfile, ...]:
    profiles: list[WheelProfile] = []
    if not profiles_dir.is_dir():
        print(f"[demo] wheel profiles dir not found: {profiles_dir}", flush=True)
        return tuple()
    for path in sorted(profiles_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        axis_map = {
            str(key): int(value) for key, value in data.get("axis_map", {}).items()
        }
        pedal = data.get("pedal", {}) or {}
        profiles.append(
            WheelProfile(
                name=str(data.get("name", path.stem)),
                display_name=str(data.get("display_name", data.get("name", path.stem))),
                detection_patterns=tuple(
                    str(pattern) for pattern in data.get("detection_patterns", ())
                ),
                axis_map=axis_map,
                inverted_pedals=bool(
                    pedal.get("inverted", data.get("inverted_pedals", True))
                ),
                invert_steering=bool(data.get("invert_steering", False)),
                ffb_enabled=bool((data.get("ffb", {}) or {}).get("enabled", False)),
                ffb_gain=float((data.get("ffb", {}) or {}).get("gain", 0.5)),
                threshold=float(data.get("threshold", 0.12)),
                is_default=bool(data.get("is_default", False)),
            )
        )
    return tuple(profiles)


def _profile_by_name(
    profiles: tuple[WheelProfile, ...], name: str
) -> WheelProfile | None:
    if name.lower() == "auto":
        return None
    normalized = name.lower().replace("_", "-")
    for profile in profiles:
        if profile.name.lower().replace("_", "-") == normalized:
            return profile
    available = ", ".join(profile.name for profile in profiles)
    raise SystemExit(
        f"Unknown wheel profile {name!r}. Available profiles: auto, {available}"
    )


def _detect_wheel(
    profiles: tuple[WheelProfile, ...],
) -> tuple[WheelProfile, Path] | None:
    # Sort default-flagged profiles to the FRONT (highest priority) so the
    # detection loop matches them before any future generic / fallback
    # profile that might overlap on the device-name pattern. ``False < True``
    # in Python, so without ``reverse=True`` the default profile would end
    # up last in the iteration order.
    ordered_profiles = sorted(
        profiles, key=lambda profile: profile.is_default, reverse=True
    )
    devices = _scan_evdev_devices()
    for profile in ordered_profiles:
        for device in devices:
            if _device_matches_profile(device, profile):
                print(
                    f"[demo] auto-detected wheel profile={profile.name} "
                    f"device={device.path} name={device.name!r}",
                    flush=True,
                )
                return profile, device.path
    if devices:
        print(
            "[demo] evdev devices seen but no wheel profile matched: "
            + ", ".join(f"{device.path}:{device.name}" for device in devices),
            flush=True,
        )
    return None


def _detect_device_for_profile(profile: WheelProfile) -> EvdevDevice | None:
    for device in _scan_evdev_devices():
        if _device_matches_profile(device, profile):
            return device
    return None


def _scan_evdev_devices() -> tuple[EvdevDevice, ...]:
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
        name = _read_evdev_name(path)
        if name is not None:
            devices.append(EvdevDevice(path=path, name=name))
    return tuple(devices)


def _device_matches_profile(device: EvdevDevice, profile: WheelProfile) -> bool:
    name = device.name.lower()
    if not any(pattern.lower() in name for pattern in profile.detection_patterns):
        return False
    required_axes = {int(axis) for axis in profile.axis_map.values()}
    return all(
        _query_axis_range(device.path, axis) is not None for axis in required_axes
    )


def _read_evdev_name(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            name_buf = array.array("B", [0] * 256)
            fcntl.ioctl(handle.fileno(), 0x80004506 + (256 << 16), name_buf)
            return name_buf.tobytes().split(b"\x00")[0].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _load_control_assets(control_assets_dir: Path | None) -> ControlAssets:
    assets_dir = control_assets_dir or Path("data/wheel_and_pedals")
    if not assets_dir.is_dir():
        if control_assets_dir is not None:
            print(
                f"[demo] control assets not found at {assets_dir}; using vector fallback",
                flush=True,
            )
        return ControlAssets(
            steering_wheel=None,
            throttle_pressed=None,
            throttle_unpressed=None,
            brake_pressed=None,
            brake_unpressed=None,
        )

    # Brake PNGs are accepted under either spelling: the AlpaSim asset
    # bundle ships them as ``break_*.png`` (a typo we inherit), but if a
    # downstream user renames them to the correct ``brake_*.png`` we
    # don't want to silently fall back to the vector renderer.
    assets = ControlAssets(
        steering_wheel=_load_asset_image(assets_dir / "steering_wheel.png"),
        throttle_pressed=_load_asset_image(assets_dir / "throttle_pressed.png"),
        throttle_unpressed=_load_asset_image(assets_dir / "throttle_unpressed.png"),
        brake_pressed=_load_first_asset_image(
            assets_dir, ("brake_pressed.png", "break_pressed.png")
        ),
        brake_unpressed=_load_first_asset_image(
            assets_dir, ("brake_unpressed.png", "break_unpressed.png")
        ),
    )
    if assets.complete:
        print(f"[demo] loaded AlpaSim control assets from {assets_dir}", flush=True)
    else:
        print(
            f"[demo] incomplete control assets at {assets_dir}; missing files use vector fallback",
            flush=True,
        )
    return assets


def _load_asset_image(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            return image.convert("RGBA").copy()
    except OSError:
        return None


def _load_first_asset_image(
    assets_dir: Path, candidate_filenames: tuple[str, ...]
) -> Image.Image | None:
    """Return the first existing asset image among the given filenames.

    Used to accept either spelling of the brake PNG (``brake_*.png`` vs
    the typo'd ``break_*.png`` shipped by AlpaSim).
    """
    for name in candidate_filenames:
        loaded = _load_asset_image(assets_dir / name)
        if loaded is not None:
            return loaded
    return None


def _apply_wheel_overrides(
    profile: WheelProfile, args: argparse.Namespace
) -> WheelProfile:
    axis_map = dict(profile.axis_map)
    if args.wheel_steering_axis is not None:
        axis_map["steering"] = int(args.wheel_steering_axis)
    if args.wheel_throttle_axis is not None:
        axis_map["throttle"] = int(args.wheel_throttle_axis)
    if args.wheel_brake_axis is not None:
        axis_map["brake"] = int(args.wheel_brake_axis)
    inverted = (
        profile.inverted_pedals
        if args.wheel_pedals_inverted is None
        else bool(args.wheel_pedals_inverted)
    )
    return WheelProfile(
        name=profile.name,
        display_name=profile.display_name,
        detection_patterns=profile.detection_patterns,
        axis_map=axis_map,
        inverted_pedals=inverted,
        invert_steering=profile.invert_steering,
        ffb_enabled=profile.ffb_enabled,
        ffb_gain=profile.ffb_gain,
        threshold=profile.threshold,
        is_default=profile.is_default,
    )


def _query_axis_range(path: Path, axis: int) -> AxisRange | None:
    try:
        with path.open("rb") as handle:
            payload = array.array("i", [0, 0, 0, 0, 0, 0])
            fcntl.ioctl(handle.fileno(), EVIOCGABS(axis), payload, True)
            return AxisRange(minimum=int(payload[1]), maximum=int(payload[2]))
    except OSError:
        return None


def _parse_axis(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected integer axis code, got {value!r}"
        ) from exc


def _tk_key_to_browser_key(keysym: str) -> str | None:
    mapping = {
        "w": "w",
        "W": "w",
        "a": "a",
        "A": "a",
        "s": "s",
        "S": "s",
        "d": "d",
        "D": "d",
        "Up": "ArrowUp",
        "Down": "ArrowDown",
        "Left": "ArrowLeft",
        "Right": "ArrowRight",
        "space": " ",
    }
    return mapping.get(keysym)


def _move_towards(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def _apply_googlemaps_filter(rgb_image: Image.Image) -> Image.Image:
    """Restyle a BEV frame to look like a Google-Maps minimap.

    The rasterizer renders lane lines / boundaries / crosswalks against a
    black background. Translate that into Google's day-mode palette by:

    1. Blending the empty (dark) regions toward a warm cream "land" tone.
    2. Blending the rendered features toward a slightly off-white "road"
       tone so they read as roads/markings instead of neon paint.

    The presence curve has a deliberate knee: anything below ~0.08
    brightness is treated as background and goes fully to land. This
    knocks down JPEG ringing around high-contrast edges (8x8 DCT blocks
    leak dim grey pixels up to ~0.10 brightness) which would otherwise
    survive a smooth-curve blend as dirty halos around vehicles and
    lane lines. The whole transform is a single numpy expression; on a
    384x384 BEV it runs in <2 ms.
    """
    # The stream loop already gave us an RGB-mode PIL Image, so skip the
    # redundant ``convert`` here; ``np.asarray`` handles the C buffer
    # directly without an extra copy.
    arr = np.asarray(rgb_image, dtype=np.float32)
    # Recolour magenta road boundaries to a low-contrast warm grey so
    # the BEV's road outlines read as Google-Maps-style soft borders
    # instead of vibrant high-contrast lines. Detection is loose on
    # purpose -- partial-coverage edge pixels (anti-aliased magenta
    # toward black/cream) get caught too, which kills the JPEG / MSAA
    # halo that was the dominant remaining aliasing offender.
    is_magenta = (
        (arr[..., 0] > 130)
        & (arr[..., 2] > 130)
        & (arr[..., 1] < arr[..., 0] * 0.55)
        & (arr[..., 1] < arr[..., 2] * 0.55)
    )
    # In-place recolour avoids the ~3 MB allocation that ``np.where``
    # would do every BEV frame at 512x512.
    np.copyto(arr, _GMAPS_BOUNDARY_GREY_FLOAT, where=is_magenta[..., np.newaxis])
    bright = arr.max(axis=2, keepdims=True) / 255.0
    # Tight knee: ``< 0.14`` brightness collapses to land, ``> 0.21``
    # is fully drawn, with only a 0.07-wide blend band so JPEG ringing
    # and bilinear-resize halos around vehicle / lane edges don't
    # survive as partial-presence grey outlines. Bilinear resampling
    # later in ``_draw_bev_panel`` adds enough natural antialiasing
    # that we don't need much soft-knee here.
    presence = np.clip((bright - 0.14) / 0.07, 0.0, 1.0)
    # Tint feature pixels toward the road colour while keeping their
    # original chroma so yellow lane paint stays warmer than white paint.
    tinted = arr * _GMAPS_TINTED_MUL
    out = tinted * presence + _GMAPS_LAND_FLOAT * (1.0 - presence)
    return Image.fromarray(out.clip(0.0, 255.0).astype(np.uint8))


def _bev_marker_y_rel() -> float:
    """Where the rig projects in the BEV image, as a fraction of height.

    Pure top-down (``BEV_TILT_DEG == 0``) puts the rig at image centre
    (0.5). Each degree of forward tilt moves it lower, by
    ``focal_y * tan(tilt) / height = tan(tilt) / (2 * tan(fov/2))``,
    which is the standard pinhole projection of a point on the rig
    plane straight below the camera.
    """
    half_fov = math.radians(BEV_FOV_DEG / 2.0)
    if half_fov <= 0:
        return 0.5
    return min(
        0.95, 0.5 + math.tan(math.radians(BEV_TILT_DEG)) / (2.0 * math.tan(half_fov))
    )


def _keyboard_drive_key(keysym: str) -> str | None:
    mapping = {
        "w": "w",
        "W": "w",
        "a": "a",
        "A": "a",
        "s": "s",
        "S": "s",
        "d": "d",
        "D": "d",
        "Up": "up",
        "Down": "down",
        "Left": "left",
        "Right": "right",
        "space": "space",
    }
    return mapping.get(keysym)


def _fit_image(image: Image.Image, bounds_wh: tuple[int, int]) -> Image.Image:
    max_w, max_h = bounds_wh
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    if size == image.size:
        # PIL's ``Image.resize`` runs ``.copy()`` on same-size input; skip it.
        return image
    return image.resize(size, Image.Resampling.BILINEAR)


if __name__ == "__main__":
    main()
