# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import argparse
import io
import math
import os
import select
import struct
import threading
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from omnidreams import scenes as _scenes
from omnidreams.interactive_drive import cli as _cli
from omnidreams.interactive_drive.app import InteractiveDriveApp
from omnidreams.interactive_drive.config import BevConfig, RasterConfig
from omnidreams.interactive_drive.input.wheel_profiles import (
    EV_ABS,
    EV_KEY,
    EVDEV_EVENT_FORMAT,
    EVDEV_EVENT_SIZE,
    AutocenterFFB,
    AxisRange,
    EvdevDevice,
    WheelProfile,
    apply_steering_curve,
    load_wheel_profiles,
    name_match_strength,
    query_axis_range,
    read_evdev_name,
    scan_evdev_devices,
    user_wheel_profiles_dir,
)
from omnidreams.scenes import normalise_scene_uuid, scenes_cache_root
from PIL import Image

# The canonical implementations of these evdev helpers now live in
# ``input/wheel_profiles.py`` so the configuration tool can share them.
# Private aliases keep the existing call sites in this module unchanged.
_scan_evdev_devices = scan_evdev_devices
_read_evdev_name = read_evdev_name
_query_axis_range = query_axis_range

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
    reverse: bool = False


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
        self._steering_range = float(profile.steering_range)
        self._steering_deadzone = float(profile.steering_deadzone)
        self._threshold = float(profile.threshold)
        self._reverse_buttons = {int(b) for b in profile.reverse_buttons}
        self._reset_buttons = {int(b) for b in profile.reset_buttons}
        self._reverse = False
        self._button_states: dict[int, int] = {}
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
            f"axes={self._profile.axis_map} ranges={self._axis_ranges} "
            f"invert_steering={self._invert_steering} "
            f"steering_range={self._steering_range} "
            f"steering_deadzone={self._steering_deadzone} "
            f"inverted_pedals={self._inverted_pedals} "
            f"reverse_buttons={sorted(self._reverse_buttons)} "
            f"reset_buttons={sorted(self._reset_buttons)}",
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
            elif event_type == EV_KEY:
                self._handle_button(int(code), int(value))

    def _handle_button(self, code: int, value: int) -> None:
        # Act on the rising edge (press) so a held button fires once.
        # Reverse toggles a sticky flag fed into every drive command; reset
        # is forwarded through the control sink, which owns the
        # KeyboardState the runtime loop reads.
        prev = self._button_states.get(code, 0)
        self._button_states[code] = value
        if value != 1 or prev == 1:
            return
        if code in self._reverse_buttons:
            self._reverse = not self._reverse
        elif code in self._reset_buttons:
            request_reset = getattr(self._control, "request_reset", None)
            if request_reset is not None:
                request_reset()

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
            self._state.reverse = self._reverse

        self._control.set_drive(
            steer=steering, throttle=throttle, brake=brake, reverse=self._reverse
        )
        self._ffb.update(abs(target_speed), gain=self._profile.ffb_gain)

    def _normalize_steering(self, raw: int) -> float:
        axis_range = self._axis_ranges[self._steering_axis]
        value = (float(raw) - axis_range.center) / (axis_range.span * 0.5)
        if self._invert_steering:
            value = -value
        return apply_steering_curve(
            value, deadzone=self._steering_deadzone, scale=self._steering_range
        )

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
        # ``speed`` is signed: positive is forward, negative is reverse. The
        # HUD shows its magnitude, so engaging reverse decelerates to 0 and
        # then builds speed in the reverse direction (rather than the digit
        # climbing forever while the throttle is held).
        direction = -1.0 if self._reverse else 1.0
        if throttle > 0.01 and brake <= 0.05:
            accel = 2.0 * throttle * dt
            current = abs(speed)
            high_speed_knee = 22.35
            if current < high_speed_knee:
                taper = max(0.2, 1.0 - (current / high_speed_knee) ** 2 * 0.5)
            else:
                excess = (current - high_speed_knee) / max(1e-6, 36.0 - high_speed_knee)
                taper = max(0.05, 0.5 * (1.0 - excess) ** 3)
            speed += direction * accel * taper
        elif brake > 0.01:
            # Brake bleeds speed toward a stop regardless of travel direction.
            speed = _move_towards(speed, 0.0, 12.0 * brake * dt)
        elif self._reverse:
            # No auto-crawl in reverse; coast toward a stop.
            speed = _move_towards(speed, 0.0, 0.5 * dt)
        else:
            creep_target = 4.47  # 10 mph, matching the AlpaSim manual-driver creep.
            if speed < creep_target + 0.1:
                # Demo crawl should be gentle: a first-order approach that
                # takes several seconds to reach 10 mph from a stop.
                speed += (creep_target - speed) * 0.18 * dt
            else:
                speed = max(0.0, speed - 0.5 * dt)
        return max(-36.0, min(36.0, speed))


def build_parser() -> argparse.ArgumentParser:
    """Build the unified ``interactive-drive`` argument parser.

    The parser is the union of three groups:

    * Backend args (``--scene``, ``--backend``, ``--manifest``,
      ``--bev``, ``--stream-mjpeg``, ...) inherited verbatim from
      :func:`omnidreams.interactive_drive.cli.build_parser`. These
      flags apply whether the user runs the supervised HUD wrapper or
      the bare backend with ``--no-hud`` / ``--stream-mjpeg``.
    * Supervisor / HUD args (``--scene-dir``, ``--autoload-scene``,
      ``--cuda-visible-devices``, ``--wheel-*``, ``--no-wheel``) that
      only matter when a HUD viewer is running. They're harmlessly
      ignored under ``--no-hud`` / ``--stream-mjpeg``.
    * The ``--no-hud`` toggle itself, which falls through to the bare
      slangpy Vulkan window. ``--stream-mjpeg`` (in the inherited
      backend group) implicitly does the same and serves the bare
      backend's frames over HTTP. For a richer browser frontend use
      ``omnidreams.webrtc.server``.
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
        " Vulkan window, or --stream-mjpeg HOST:PORT to skip the local"
        " window entirely and serve frames to a browser as an MJPEG"
        " HTTP stream (useful on compute-only hosts without a Vulkan"
        " GPU). For a richer browser viewer use the separate"
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
    # ``--stream-mjpeg`` runs through ``_run_streaming`` so the long-lived
    # MJPEG presenter (HTTP server, browser session) survives across
    # scene-change requests posted by the in-page picker. ``--no-hud``
    # without MJPEG drops straight through to the bare CLI's Vulkan
    # window, which has no scene picker UI of its own. The default path
    # is the slangpy HUD with full chrome.
    if args.stream_mjpeg is not None:
        _run_streaming(args)
        return
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
                " (typo? expected a path or bundled config name like "
                "example_world_model.yaml)"
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
    # Attach the wheel up front, bound to the placeholder keyboard, so the
    # HUD's steering / pedal chrome reacts to the physical device during the
    # initial "Load Scene" wait -- not only once a scene is running. The
    # evdev reader thread starts now; the per-run factory below just rebinds
    # its drive sink to each run's KeyboardState.
    wheel: Any = None
    if wheel_selection is not None:
        profile, device_path = wheel_selection
        wheel = WheelBridge(
            device_path=device_path,
            profile=profile,
            control=KeyboardStateDriveSink(placeholder_keyboard),
        )
        wheel.start()
        presenter.set_wheel(wheel)

    def _factory(config: Any, keyboard: Any) -> Any:
        # Called once per ``InteractiveDriveApp.__init__``. Rebind the
        # wheel's drive sink to this run's KeyboardState (the reader thread
        # started above keeps running, state-machine-clean -- the only thing
        # tied to the keyboard is the sink it posts ``set_drive`` into), then
        # point the presenter at the new keyboard.
        if wheel is not None:
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


def _run_streaming(args: argparse.Namespace) -> None:
    """Run the engine with the MJPEG streaming presenter and a scene-change loop.

    Mirrors :func:`_run_slangpy_hud`'s outer-loop structure but with a
    long-lived :class:`MJPEGStreamingPresenter` instead of a slangpy
    window. The HTTP server (and any connected browser sessions) stay
    alive across scene transitions; only the backend / pipeline /
    simulation get rebuilt per scene. The browser shows the loading
    overlay during the rebuild and resumes streaming the moment the
    new pipeline produces its first chunk.

    Scene options come from the same discovery layer the slangpy HUD
    uses. They get serialised into a JSON-friendly shape and posted to
    the presenter so the in-browser ``/scenes`` endpoint can populate
    its dropdown without round-tripping through the SceneOption class.
    """
    from omnidreams.interactive_drive.input.keyboard import KeyboardState
    from omnidreams.interactive_drive.streaming_presenter import (
        MJPEGStreamingPresenter,
        parse_bind,
    )

    _apply_cuda_visible_devices_inplace(args.cuda_visible_devices)
    _resolve_demo_paths(args)
    scene_options = _discover_scene_options(args.scene_dir, args.scene)
    if not args.scene.exists() and scene_options:
        args.scene = scene_options[0].path
    if args.backend == "omnidreams":
        if args.manifest is None:
            raise SystemExit("--manifest is required for the omnidreams backend")
        if not args.manifest.exists():
            raise SystemExit(
                f"--manifest path does not exist: {args.manifest}"
                " (typo? expected a path or bundled config name like "
                "example_world_model.yaml)"
            )
    if not scene_options and not args.scene.exists():
        raise SystemExit(
            f"--scene path does not exist and --scene-dir contains no scenes: {args.scene}"
        )

    # JSON-serialisable form of the discovered scenes for the browser
    # ``/scenes`` endpoint. Thumbnails are JPEG-encoded once at startup
    # and stashed on the presenter so the per-card ``/thumbnail``
    # request just blobs the bytes back -- no per-request encode cost
    # under the HTTP handler thread, which would otherwise compete
    # with the main camera's encode budget.
    scenes_payload: tuple[dict[str, object], ...] = tuple(
        {
            "label": opt.label,
            "path": str(opt.path),
            "variants": list(opt.variants),
        }
        for opt in scene_options
    )
    thumbnails: dict[str, bytes] = {}
    for opt in scene_options:
        if opt.thumbnail is None:
            continue
        buf = io.BytesIO()
        # PIL's RGBA / palette-mode thumbnails need an explicit RGB
        # conversion before JPEG encode. The discovery layer already
        # returns RGB, but be defensive in case it changes upstream.
        thumb_rgb = (
            opt.thumbnail
            if opt.thumbnail.mode == "RGB"
            else opt.thumbnail.convert("RGB")
        )
        thumb_rgb.save(buf, format="JPEG", quality=85)
        thumbnails[str(opt.path)] = buf.getvalue()

    bind_host, bind_port = parse_bind(args.stream_mjpeg)
    placeholder_keyboard = KeyboardState()
    presenter = MJPEGStreamingPresenter(
        raster=RasterConfig(),
        keyboard=placeholder_keyboard,
        bind_host=bind_host,
        bind_port=bind_port,
        scenes=scenes_payload,
        thumbnails=thumbnails,
    )

    def _factory(config: object, keyboard: KeyboardState) -> MJPEGStreamingPresenter:
        # Each ``InteractiveDriveApp.__init__`` builds a fresh
        # ``KeyboardState``; rebind so the long-lived presenter
        # follows the new instance instead of writing into the
        # placeholder keyboard from before the first scene loaded.
        del config
        presenter.bind_keyboard(keyboard)
        return presenter

    try:
        # Don't auto-load: always wait for the browser to pick the
        # first scene. This mirrors the slangpy HUD's ``--no-autoload-
        # scene`` default (which is the *only* mode for the streaming
        # path -- there's no Vulkan window to show progress in, so we
        # would otherwise burn world-model warmup on whatever
        # ``args.scene`` defaulted to before the user expressed any
        # intent). The presenter publishes an idle "Select a scene to
        # begin" overlay frame so connected browsers have something to
        # render while the wait spins.
        print(
            "[demo] streaming presenter waiting for first scene selection...",
            flush=True,
        )
        request = presenter.wait_for_scene_selection()
        if request is None:
            return  # presenter closed before any selection (Ctrl-C)
        first_scene_path, first_variant = request
        args.scene = first_scene_path
        args.variant = first_variant
        presenter.acknowledge_scene_change(first_scene_path, first_variant)
        print(
            f"[demo] streaming initial scene -> {first_scene_path.name} "
            f"variant={first_variant!r}",
            flush=True,
        )

        while True:
            config, backend = _cli.prepare_config_and_backend(args)
            app = InteractiveDriveApp(
                config=config,
                backend=backend,
                presenter_factory=_factory,
                close_presenter_on_exit=False,
            )
            app.run()
            requested = presenter.pending_scene_change
            if requested is None:
                # Either the process is shutting down (Ctrl-C) or the
                # rollout finished without a scene-change request.
                # ``MJPEGStreamingPresenter`` has no native quit
                # affordance, so a "no pending change" exit is
                # treated as the end of the session.
                break
            new_scene_path, new_variant = requested
            args.scene = new_scene_path
            args.variant = new_variant
            presenter.acknowledge_scene_change(new_scene_path, new_variant)
            print(
                f"[demo] streaming scene change -> {new_scene_path.name} "
                f"variant={new_variant!r}",
                flush=True,
            )
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
    for attr in ("scene", "scene_dir", "wheel_profiles_dir"):
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, _project_path(value))
    if args.manifest is not None:
        args.manifest = _cli.resolve_manifest_path(args.manifest)
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
# :func:`omnidreams.scenes.variant_from_stem` -- all three discovery paths
# (USDZ archives, unpacked scene dirs, HUD variant selector) now agree on
# both canonical ``prompt_<N>.txt`` names and legacy numeric
# ``prompt<N>.txt`` names.


def _variant_label(variant: str) -> str:
    labels = {
        "default": "Default",
        "1": "Bright Midday Sun",
        "2": "Snowstorm",
        "3": "Night with Heavy Rain",
    }
    return labels.get(variant, variant)


def _merged_wheel_profiles(cli_profiles_dir: Path) -> tuple[WheelProfile, ...]:
    """Profiles from the user config dir plus any ``--wheel-profiles-dir``.

    User-generated profiles (written by ``interactive-drive-configuration``)
    live in :func:`user_wheel_profiles_dir` and take precedence over a
    profile of the same name found in the CLI-provided directory.
    """
    merged: dict[str, WheelProfile] = {}
    for profile in (
        *load_wheel_profiles(user_wheel_profiles_dir()),
        *load_wheel_profiles(cli_profiles_dir),
    ):
        merged.setdefault(profile.name.lower(), profile)
    return tuple(merged.values())


def _select_wheel(args: argparse.Namespace) -> tuple[WheelProfile, Path] | None:
    profiles = _merged_wheel_profiles(args.wheel_profiles_dir)
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
    best: tuple[int, WheelProfile] | None = None
    for profile in ordered:
        strength = _profile_device_match_strength(fake_device, profile)
        if strength > 0 and (best is None or strength > best[0]):
            best = (strength, profile)
    return best[1] if best is not None else None


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
        device = _best_device_for_profile(profile, devices)
        if device is not None:
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
    return _best_device_for_profile(profile, _scan_evdev_devices())


def _profile_device_match_strength(device: EvdevDevice, profile: WheelProfile) -> int:
    """Match score for *device* vs *profile*: 0 none, 1 substring, 2 exact name.

    A non-zero score also requires every axis in the profile's ``axis_map``
    to exist on the device. The exact-name tier is what stops a profile
    captured from e.g. ``"Wireless Controller"`` from binding a sibling node
    (``"Wireless Controller Motion Sensors"``) whose name merely contains the
    pattern and may even expose overlapping axes.
    """
    if not profile.detection_patterns:
        return 0
    required_axes = {int(axis) for axis in profile.axis_map.values()}
    if not all(
        _query_axis_range(device.path, axis) is not None for axis in required_axes
    ):
        return 0
    return name_match_strength(device.name, profile.detection_patterns)


def _best_device_for_profile(
    profile: WheelProfile, devices: tuple[EvdevDevice, ...]
) -> EvdevDevice | None:
    """Return the device matching *profile* best, preferring an exact name.

    Among all connected devices an exact-name match always beats a substring
    match, so the real controller node wins over a same-named sensor/touchpad
    sibling regardless of scan order.
    """
    best: tuple[int, EvdevDevice] | None = None
    for device in devices:
        strength = _profile_device_match_strength(device, profile)
        if strength > 0 and (best is None or strength > best[0]):
            best = (strength, device)
    return best[1] if best is not None else None


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
    # Use ``replace`` so every other field is preserved. The previous manual
    # reconstruction silently dropped any field it didn't relist -- which
    # reset steering_range / steering_deadzone to their no-op defaults and
    # unbound reverse/reset buttons at runtime, even though the saved
    # profile had them.
    return replace(profile, axis_map=axis_map, inverted_pedals=inverted)


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
