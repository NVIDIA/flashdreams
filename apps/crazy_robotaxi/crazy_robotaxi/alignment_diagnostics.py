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

"""Frame-synchronized alignment diagnostics for Crazy Robotaxi."""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omnidreams_game_engine.math3d import extract_yaw_from_transform
from omnidreams_game_engine.types import (
    CameraCalibration,
    PhysicsDebugFrame,
    PresentedFrame,
    VehicleState,
)
from PIL import Image, ImageDraw, ImageFont

_PANEL_HEIGHT = 360
"""Height of each image panel in a diagnostic contact sheet."""

_PHYSX_RADIUS_M = 45.0
"""World radius shown around the ego in the PhysX topology panel."""


@dataclass(frozen=True)
class _CapturedFrame:
    sequence: int
    timestamp_us: int
    condition_rgb: np.ndarray
    generated_rgb: np.ndarray
    bev_rgb: np.ndarray | None
    physx_rgb: np.ndarray | None
    telemetry: dict[str, object]


class AlignmentDiagnosticRecorder:
    """Persist synchronized model inputs, outputs, physics, and poses."""

    def __init__(self, output_root: Path) -> None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.output_dir = (
            output_root.expanduser().resolve() / f"run-{run_id}-{os.getpid()}"
        )
        self._frames_dir = self.output_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=False)
        self._queue: queue.Queue[_CapturedFrame | None] = queue.Queue(maxsize=16)
        self._writer_error: BaseException | None = None
        self._closed = False
        self._sequence = 0
        self._last_frame_identity: tuple[int, int] | None = None
        self._metadata: dict[str, object] = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "panels": [
                "HD-map conditioning",
                "generated RGB",
                "BEV",
                "PhysX contact pose",
            ],
        }
        self._write_metadata()
        self._writer = threading.Thread(
            target=self._write_frames,
            name="taxi-alignment-diagnostics",
            daemon=True,
        )
        self._writer.start()

    def configure_camera(self, calibration: CameraCalibration) -> None:
        """Record the camera calibration used by subsequent frames."""
        self._metadata["camera"] = {
            "clipgt_name": calibration.clipgt_name,
            "logical_name": calibration.logical_name,
            "width": calibration.width,
            "height": calibration.height,
            "cx": calibration.cx,
            "cy": calibration.cy,
            "polynomial": calibration.polynomial.tolist(),
            "is_backward_polynomial": calibration.is_backward_polynomial,
            "linear_cde": calibration.linear_cde.tolist(),
            "sensor_to_rig_flu": calibration.sensor_to_rig_flu.tolist(),
        }
        self._write_metadata()

    def record_scene(self, scene_path: object, variant: str) -> None:
        """Record the scene selection associated with subsequent frames."""
        self._metadata["scene_path"] = str(scene_path)
        self._metadata["variant"] = variant
        self._write_metadata()

    def capture(self, frame: PresentedFrame) -> None:
        """Queue one fully synchronized generated frame for persistence."""
        if self._closed or self._writer_error is not None:
            if self._writer_error is not None:
                raise RuntimeError(
                    "Taxi alignment diagnostic writer failed"
                ) from self._writer_error
            return
        if (
            frame.vehicle_state is None
            or frame.rig_to_world is None
            or frame.model_rgb_host_uint8 is None
        ):
            return
        identity = (id(frame), int(frame.timestamp_us))
        if identity == self._last_frame_identity:
            return
        self._last_frame_identity = identity

        condition_rgb = _materialize_rgb(frame.rgb_host_uint8)
        generated_rgb = _materialize_rgb(frame.model_rgb_host_uint8)
        bev_rgb = (
            None
            if frame.bev_host_uint8 is None
            else _materialize_rgb(frame.bev_host_uint8)
        )
        physx_rgb = _render_physx_topdown(frame.physx_debug, frame.vehicle_state)
        telemetry = _frame_telemetry(frame, self._sequence)
        captured = _CapturedFrame(
            sequence=self._sequence,
            timestamp_us=int(frame.timestamp_us),
            condition_rgb=condition_rgb,
            generated_rgb=generated_rgb,
            bev_rgb=bev_rgb,
            physx_rgb=physx_rgb,
            telemetry=telemetry,
        )
        self._sequence += 1
        self._queue.put(captured)

    def close(self) -> None:
        """Flush queued frames and close the diagnostic artifact."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._writer.join()
        self._metadata["frame_count"] = self._sequence
        self._metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_metadata()
        if self._writer_error is not None:
            raise RuntimeError(
                "Taxi alignment diagnostic writer failed"
            ) from self._writer_error

    def _write_metadata(self) -> None:
        path = self.output_dir / "metadata.json"
        path.write_text(json.dumps(self._metadata, indent=2) + "\n", encoding="utf-8")

    def _write_frames(self) -> None:
        telemetry_path = self.output_dir / "telemetry.csv"
        try:
            with telemetry_path.open("w", newline="", encoding="utf-8") as handle:
                writer: csv.DictWriter[str] | None = None
                while True:
                    captured = self._queue.get()
                    if captured is None:
                        break
                    if writer is None:
                        writer = csv.DictWriter(
                            handle, fieldnames=list(captured.telemetry)
                        )
                        writer.writeheader()
                    writer.writerow(captured.telemetry)
                    handle.flush()
                    contact_sheet = _build_contact_sheet(captured)
                    contact_sheet.save(
                        self._frames_dir / f"frame_{captured.sequence:06d}.png",
                        format="PNG",
                    )
        except BaseException as exc:  # noqa: BLE001 - re-raised by the owner thread
            self._writer_error = exc


class AlignmentDiagnosticPresenter:
    """Record synchronized frames before HUD overlays are applied."""

    def __init__(self, presenter: Any, output_root: Path) -> None:
        self._presenter = presenter
        self._recorder = AlignmentDiagnosticRecorder(output_root)

    @property
    def output_dir(self) -> Path:
        """Return the timestamped directory receiving diagnostic artifacts."""
        return self._recorder.output_dir

    def configure_taxi_camera(self, calibration: CameraCalibration) -> None:
        """Record and forward the active Taxi camera calibration."""
        self._recorder.configure_camera(calibration)
        configure = getattr(self._presenter, "configure_taxi_camera", None)
        if callable(configure):
            configure(calibration)

    def acknowledge_scene_change(self, scene_path: object, variant: str) -> Any:
        """Record and forward a selected scene."""
        self._recorder.record_scene(scene_path, variant)
        return self._presenter.acknowledge_scene_change(scene_path, variant)

    def present_frame(self, frame: PresentedFrame, view_mode: str) -> None:
        """Capture synchronized inputs before presenting the frame."""
        self._recorder.capture(frame)
        self._presenter.present_frame(frame, view_mode=view_mode)

    def close(self) -> None:
        """Flush the recorder and close the wrapped presenter."""
        try:
            self._recorder.close()
        finally:
            self._presenter.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._presenter, name)


def _materialize_rgb(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Diagnostic RGB frame must be HWC RGB(A), got {array.shape}")
    if array.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(array.dtype, np.floating) else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(array[..., :3]).copy()


def _frame_telemetry(frame: PresentedFrame, sequence: int) -> dict[str, object]:
    state = frame.vehicle_state
    rig = frame.rig_to_world
    assert state is not None and rig is not None
    rig_yaw = extract_yaw_from_transform(rig)
    debug = frame.physx_debug
    physx_yaw = (
        math.nan
        if debug is None
        else _yaw_from_quaternion_xyzw(debug.ego_orientation_xyzw)
    )
    physx_position = (
        np.full(3, np.nan, dtype=np.float32) if debug is None else debug.ego_position_m
    )
    command = frame.driver_command
    motion = frame.model_motion_metrics or {}
    return {
        "sequence": sequence,
        "timestamp_us": int(frame.timestamp_us),
        "x_m": state.x_m,
        "y_m": state.y_m,
        "z_m": state.z_m,
        "yaw_rad": state.yaw_rad,
        "pitch_rad": state.pitch_rad,
        "roll_rad": state.roll_rad,
        "speed_mps": state.speed_mps,
        "steer_rad": state.steer_rad,
        "yaw_rate_radps": state.yaw_rate_radps,
        "velocity_x_mps": state.velocity_x_mps,
        "velocity_y_mps": state.velocity_y_mps,
        "rig_x_m": float(rig[0, 3]),
        "rig_y_m": float(rig[1, 3]),
        "rig_z_m": float(rig[2, 3]),
        "rig_yaw_rad": rig_yaw,
        "state_rig_yaw_error_rad": _angle_delta(state.yaw_rad, rig_yaw),
        "physx_x_m": float(physx_position[0]),
        "physx_y_m": float(physx_position[1]),
        "physx_z_m": float(physx_position[2]),
        "physx_yaw_rad": physx_yaw,
        "state_physx_yaw_error_rad": _angle_delta(state.yaw_rad, physx_yaw),
        "state_physx_xy_error_m": math.hypot(
            state.x_m - float(physx_position[0]),
            state.y_m - float(physx_position[1]),
        ),
        "command_throttle": math.nan if command is None else command.throttle,
        "command_brake": math.nan if command is None else command.brake,
        "command_steer": math.nan if command is None else command.steer,
        "command_reverse": False if command is None else command.reverse,
        "impact_kind": frame.impact_kind or "",
        "model_view_fallback_reason": frame.model_view_fallback_reason or "",
        "motion_axis": motion.get("axis", ""),
        "motion_mismatched": motion.get("mismatched", False),
        "condition_motion_px": motion.get("condition_component_px", math.nan),
        "generated_motion_px": motion.get("generated_component_px", math.nan),
        "motion_check_ms": motion.get("elapsed_ms", math.nan),
    }


def _build_contact_sheet(captured: _CapturedFrame) -> Image.Image:
    panels = [
        ("HD-MAP CONDITIONING", captured.condition_rgb),
        ("GENERATED RGB", captured.generated_rgb),
        ("BEV", captured.bev_rgb),
        ("PHYSX CONTACT POSE", captured.physx_rgb),
    ]
    rendered = [_render_panel(label, rgb) for label, rgb in panels]
    width = sum(panel.width for panel in rendered)
    header_height = 42
    canvas = Image.new("RGB", (width, _PANEL_HEIGHT + header_height), (12, 12, 18))
    x_px = 0
    for panel in rendered:
        canvas.paste(panel, (x_px, header_height))
        x_px += panel.width
    draw = ImageDraw.Draw(canvas)
    telemetry = captured.telemetry
    draw.text(
        (10, 6),
        (
            f"frame={captured.sequence} timestamp_us={captured.timestamp_us} "
            f"position=({float(telemetry['x_m']):.2f}, {float(telemetry['y_m']):.2f}) "
            f"yaw={float(telemetry['yaw_rad']):+.4f} "
            f"speed={float(telemetry['speed_mps']):+.2f}m/s "
            f"state/rig yaw error={float(telemetry['state_rig_yaw_error_rad']):+.6f}"
        ),
        fill=(235, 235, 240),
        font=ImageFont.load_default(),
    )
    return canvas


def _render_panel(label: str, rgb: np.ndarray | None) -> Image.Image:
    if rgb is None:
        panel = Image.new("RGB", (640, _PANEL_HEIGHT), (24, 24, 30))
    else:
        image = Image.fromarray(rgb, mode="RGB")
        width = max(1, round(image.width * _PANEL_HEIGHT / image.height))
        panel = image.resize((width, _PANEL_HEIGHT), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 24), fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255), font=ImageFont.load_default())
    return panel


def _render_physx_topdown(
    debug: PhysicsDebugFrame | None, state: VehicleState
) -> np.ndarray | None:
    if debug is None:
        return None
    size = _PANEL_HEIGHT
    image = Image.new("RGB", (size, size), (28, 28, 32))
    draw = ImageDraw.Draw(image)
    scale = size / (2.0 * _PHYSX_RADIUS_M)

    def project(point_xy: np.ndarray) -> tuple[float, float]:
        delta = point_xy - np.asarray([state.x_m, state.y_m], dtype=np.float32)
        cos_yaw = math.cos(state.yaw_rad)
        sin_yaw = math.sin(state.yaw_rad)
        forward_m = cos_yaw * float(delta[0]) + sin_yaw * float(delta[1])
        left_m = -sin_yaw * float(delta[0]) + cos_yaw * float(delta[1])
        return size * 0.5 - left_m * scale, size * 0.5 - forward_m * scale

    for segment in debug.barrier_segments_xy_m:
        draw.line(
            (project(segment[0]), project(segment[1])), fill=(255, 210, 70), width=3
        )
    for position, dimensions in zip(
        debug.actor_positions_m,
        debug.actor_dimensions_lwh,
        strict=True,
    ):
        center_x, center_y = project(position[:2])
        radius = max(2.0, float(max(dimensions[:2])) * scale * 0.5)
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            outline=(220, 80, 80),
            width=2,
        )
    contact_yaw = _yaw_from_quaternion_xyzw(debug.ego_orientation_xyzw)
    contact_forward = np.asarray(
        [math.cos(contact_yaw), math.sin(contact_yaw)], dtype=np.float32
    )
    contact_left = np.asarray(
        [-contact_forward[1], contact_forward[0]], dtype=np.float32
    )
    half_length_m = float(debug.ego_dimensions_lwh[0]) * 0.5
    half_width_m = float(debug.ego_dimensions_lwh[1]) * 0.5
    contact_xy = debug.ego_position_m[:2]
    contact_corners = [
        contact_xy
        + forward_sign * half_length_m * contact_forward
        + left_sign * half_width_m * contact_left
        for forward_sign, left_sign in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]
    draw.polygon(
        [project(corner) for corner in contact_corners],
        fill=(118, 185, 0),
        outline=(255, 255, 255),
    )
    app_center = project(np.asarray([state.x_m, state.y_m], dtype=np.float32))
    draw.ellipse(
        (
            app_center[0] - 3,
            app_center[1] - 3,
            app_center[0] + 3,
            app_center[1] + 3,
        ),
        fill=(60, 180, 255),
    )
    return np.asarray(image)


def _yaw_from_quaternion_xyzw(quaternion: np.ndarray) -> float:
    x, y, z, w = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _angle_delta(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return math.nan
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi
