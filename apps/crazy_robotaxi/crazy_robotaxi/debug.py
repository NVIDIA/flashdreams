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

"""Headless game observations through an app-owned FlashDreams output mode."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import socket
import sys
import time
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, TextIO, cast

import numpy as np
import torch
from omnidreams_game_engine.math3d import quaternion_to_matrix_xyzw
from PIL import Image, ImageDraw, ImageFont

from crazy_robotaxi.debug_input import DEBUG_CONTROL_KEY, HeadlessInput
from crazy_robotaxi.ui import CrazyRobotaxiImGuiUILoop, TaxiHudState
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IUILoop, ModelInferenceState
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_encoder import Mp4Encoder, result_to_rgb24_tensor
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

DEBUG_METADATA_KEY = "robotaxi_debug"
"""Session metadata describing the selected map and driving script."""


def render_physics_view(
    record: dict[str, Any],
    width: int,
    height: int,
    trail_xy: list[tuple[float, float]],
) -> np.ndarray:
    """Draw a world-aligned overhead view of measured bounds and curb response.

    Use the pre-policy physics pose rather than the ground-aligned camera pose.
    White outlines show the game's contact bounds; blue outlines show the native
    inset chassis. The curb-response flag includes the game's proximity rebound.
    """
    colliders = record["physics_colliders"]
    position = np.asarray(colliders["ego_position_m"], dtype=float)
    scale = min(width, height) / 32.0
    image = Image.new("RGB", (width, height), (18, 22, 30))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, min(width, height) / 32))

    def pixels(points: Any) -> list[tuple[float, float]]:
        xy = np.asarray(points)[..., :2] - position[:2]
        return list(
            zip(width / 2 + xy[..., 0] * scale, height / 2 - xy[..., 1] * scale)
        )

    for axis in range(2):
        extent = (width if axis == 0 else height) / scale / 2
        for coordinate in np.arange(
            math.floor((position[axis] - extent) / 2) * 2,
            position[axis] + extent,
            2,
        ):
            endpoints = np.repeat(position[None, :], 2, axis=0)
            endpoints[:, axis] = coordinate
            endpoints[:, 1 - axis] += [-1000, 1000]
            draw.line(pixels(endpoints), fill=(32, 38, 48))
    segments = np.asarray(colliders["barrier_segments_xy_m"])
    for segment, thickness in zip(
        segments, colliders["barrier_thicknesses_m"], strict=True
    ):
        draw.line(
            pixels(segment), fill=(0, 220, 235), width=max(2, round(thickness * scale))
        )
    if len(trail_xy) > 1:
        draw.line(pixels(trail_xy), fill=(130, 130, 145), width=2)

    def box(
        center: Any, quaternion: Any, dimensions: Any, color: tuple[int, int, int]
    ) -> None:
        rotation = quaternion_to_matrix_xyzw(np.asarray(quaternion).tolist())
        signs = np.asarray(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        )
        corners = signs * np.asarray(dimensions) / 2 @ rotation.T + center
        projected = pixels(corners)
        for i in range(8):
            for bit in (1, 2, 4):
                if not i & bit:
                    draw.line([projected[i], projected[i | bit]], fill=color, width=2)

    for center, quaternion, dimensions in zip(
        colliders["actor_positions_m"],
        colliders["actor_orientations_xyzw"],
        colliders["actor_dimensions_lwh"],
        strict=True,
    ):
        box(center, quaternion, dimensions, (255, 185, 70))
    orientation = np.asarray(colliders["ego_orientation_xyzw"])
    rotation = quaternion_to_matrix_xyzw(orientation.tolist())
    box(position, orientation, colliders["ego_dimensions_lwh"], (245, 245, 245))
    if colliders["ego_chassis_dimensions_lwh"] is not None:
        center = position + rotation @ np.asarray(colliders["ego_chassis_offset_m"])
        box(
            center, orientation, colliders["ego_chassis_dimensions_lwh"], (70, 150, 255)
        )
    draw.line(
        pixels([position, position + rotation[:, 0] * 4]), fill=(110, 240, 110), width=3
    )
    physical = record["physics_vehicle"]
    contact = record["static_barrier_collision"]
    color = (255, 90, 70) if contact else (245, 245, 245)
    draw.rectangle((0, 0, width, min(height, 66)), fill=(18, 22, 30))
    draw.text(
        (8, 8),
        f"PHYSICS  frame {record['frame']}  {record['frame'] / 30:.2f}s"
        f"  {physical['speed_mps']:.1f} m/s  CURB RESPONSE: {'YES' if contact else 'no'}",
        fill=color,
        font=font,
    )
    draw.text(
        (8, 26),
        f"z {physical['z_m']:.3f} m  roll {math.degrees(physical['roll_rad']):.1f} deg"
        f"  pitch {math.degrees(physical['pitch_rad']):.1f} deg",
        fill=(245, 245, 245),
        font=font,
    )
    draw.text(
        (8, 44),
        "Cyan: curbs | White: contact bounds | Blue: chassis | Green: heading",
        fill=(245, 245, 245),
        font=font,
    )
    return np.asarray(image)


@dataclass(frozen=True, slots=True, match_args=False)
class DebugFrameResult(StepResult):
    """A generated/HD-map frame pair with its aligned simulation record."""

    record: dict[str, Any] | None = None


class RobotaxiDebugUILoop(IUILoop[TaxiHudState]):
    """Present every generated/HD-map pair without a graphics window or HUD."""

    def _initialize_loop_state(self) -> None:
        self._last_presented_count = 0

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Package both model channels and their matching telemetry for the sink."""
        del events
        count = self.presented_model_frame_count
        if count == self._last_presented_count:
            return []
        self._last_presented_count = count
        frames = self.presented_model_frames()
        if len(frames) < 2:
            # Startup menu frames carry no generated/conditioning pair.
            return []
        hud_frame = self.state.select_presented_frame(frames[0])
        if hud_frame is None or hud_frame.frame_key != int(frames[0].data_ptr()):
            raise RuntimeError("Debug frame has no aligned game telemetry")
        if hud_frame.debug_record is None:
            raise RuntimeError("Paired capture requires physics observations")
        # The public UI protocol carries one result. Preserve both channels in
        # that result; the app-owned sink separates them before writing files.
        return [
            DebugFrameResult(
                step_index=step_index,
                output=torch.cat(frames[:2], dim=-1).unsqueeze(0),
                frame_count=1,
                output_layout=VideoTensorLayout.tchw,
                record=hud_frame.debug_record,
            )
        ]

    def is_finished(self) -> bool:
        return (
            self.model_inference_state is ModelInferenceState.FINISHED
            and not self.has_pending_model_frames()
            and self._last_presented_count == self.presented_model_frame_count
        )

    def reset(self) -> None:
        """Clear presentation and game state for a new run."""
        self._last_presented_count = 0
        self.state.reset()


class RobotaxiLiveDebugUILoop(CrazyRobotaxiImGuiUILoop):
    """Run the ordinary interactive UI offscreen until the capture ends."""

    def is_finished(self) -> bool:
        return super().is_finished() or (
            self.model_inference_state is ModelInferenceState.FINISHED
            and not self.has_pending_model_frames()
        )


class RobotaxiDebugWindow(IClientWindow):
    """Write synchronized views, contact PNGs, and aligned JSONL observations."""

    def __init__(
        self,
        directory: Path,
        *,
        snapshot_every: int = 30,
        cpu_sample_ms: float = 0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.directory = directory
        self.snapshot_every = snapshot_every
        self.frame_count = 0
        self._desc: SessionDesc | None = None
        self._encoders: dict[str, Mp4Encoder] = {}
        self._telemetry: TextIO | None = None
        self._finalized = False
        self._physics_trail: deque[tuple[float, float]] = deque(maxlen=90)
        self._contact_frames: list[int] = []
        self._last_frames: dict[str, np.ndarray] = {}
        self._input_events: list[UserInputEvent] = []
        self._clock_ns = clock_ns
        self._started_ns = 0
        self._paired_views = False
        self.control = HeadlessInput(clock_ns)
        self._model_steps: TextIO | None = None
        self._cpu_samples: TextIO | None = None
        self._cpu_sample_ns = int(cpu_sample_ms * 1_000_000)
        self._last_sample_ns = 0
        self._last_simulation_frame = (-1, -1)
        self._control_socket: socket.socket | None = None
        self._input_log: TextIO | None = None
        self._presentation_size: tuple[int, int] | None = None

    def open(self, session_desc: SessionDesc) -> None:
        """Start one capture in a fresh directory using the game session contract."""
        if DEBUG_METADATA_KEY not in session_desc.metadata:
            raise ValueError(
                "--mode robotaxi-debug requires Crazy Robotaxi capture metadata"
            )
        metadata = session_desc.metadata[DEBUG_METADATA_KEY]
        self._paired_views = metadata.get("paired_views", False)
        self.control = session_desc.metadata.get(DEBUG_CONTROL_KEY, self.control)
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Debug capture requires tchw output")
        if session_desc.video_width % 2 or session_desc.video_height % 2:
            raise ValueError("Debug videos require even pixel dimensions")
        self.directory.mkdir(parents=True, exist_ok=False)
        self._desc = session_desc
        metadata = session_desc.metadata[DEBUG_METADATA_KEY]
        if "map_source" in metadata:
            inputs = self.directory / "inputs"
            inputs.mkdir()
            (inputs / Path(metadata["map"]).name).write_text(
                metadata["map_source"], encoding="utf-8"
            )
            if metadata.get("script_events"):
                (inputs / "drive.yaml").write_text(
                    json.dumps({"events": metadata["script_events"]}, indent=2) + "\n",
                    encoding="utf-8",
                )
            (inputs / "settings.yaml").write_text(
                json.dumps(metadata["settings"], indent=2) + "\n",
                encoding="utf-8",
            )
        self._telemetry = (self.directory / "telemetry.jsonl").open(
            "x", encoding="utf-8"
        )
        self._started_ns = self._clock_ns()
        self._input_log = (self.directory / "input_events.jsonl").open(
            "x", encoding="utf-8"
        )
        self._model_steps = (self.directory / "model_steps.jsonl").open(
            "x", encoding="utf-8"
        )
        if self._cpu_sample_ns:
            self._cpu_samples = (self.directory / "cpu_samples.jsonl").open(
                "x", encoding="utf-8"
            )
        self._control_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._control_socket.setblocking(False)
        self._control_socket.bind(str(self.directory / "control.sock"))
        for command in metadata.get("script_events", []):
            self.control.accept(command)
        self._flush_control()
        if not self._paired_views:
            (self.directory / "frames" / "presentation").mkdir(parents=True)
        for name in ("generated", "hdmap", "physics") if self._paired_views else ():
            (self.directory / "frames" / name).mkdir(parents=True)
            self._encoders[name] = Mp4Encoder(
                self.directory / f"{name}.mp4",
                width=session_desc.video_width,
                height=session_desc.video_height,
                frames_per_second=session_desc.frames_per_second_for_step,
            )
        self._write_manifest()

    def get_user_input_events(self) -> UserInputEvents:
        """Deliver due agent/file commands through the ordinary runtime buffer."""
        if self._control_socket is not None:
            # Bound each poll so a producer cannot starve presentation.
            for _ in range(64):
                try:
                    packet, peer = self._control_socket.recvfrom(4096)
                except BlockingIOError:
                    break
                try:
                    payload = json.loads(packet)
                except (ValueError, UnicodeError, RecursionError):
                    self.control.messages.put(
                        (peer or None, {"status": "rejected", "reason": "invalid_json"})
                    )
                else:
                    self.control.accept(payload, peer or None)
        events, self._input_events = self._input_events, []
        events.extend(self.control.due())
        self._sample_model_thread()
        self._flush_control()
        return UserInputEvents(events)

    def put_user_input_event(self, event: UserInputEvent) -> None:
        """Retain framework lifecycle events without changing their timestamps."""
        self._input_events.append(event)

    def _flush_control(self) -> None:
        while True:
            try:
                peer, message = self.control.messages.get_nowait()
            except Empty:
                break
            encoded = json.dumps(message, default=_json_value, allow_nan=False)
            log = (
                self._model_steps
                if message["status"] == "model_step"
                else self._input_log
            )
            if log is not None:
                log.write(encoded + "\n")
                log.flush()
            if peer and self._control_socket is not None:
                try:
                    self._control_socket.sendto(encoded.encode(), peer)
                except OSError:
                    # A disconnected/full agent socket must not stall gameplay.
                    # The durable JSONL log retains every acknowledgment.
                    pass

    def _sample_model_thread(self) -> None:
        if self._cpu_samples is None or self.control.model_ident is None:
            return
        now = self._clock_ns()
        if now - self._last_sample_ns < self._cpu_sample_ns:
            return
        self._last_sample_ns = now
        frame = sys._current_frames().get(self.control.model_ident)
        stack = []
        while frame is not None:
            stack.append(
                {
                    "file": frame.f_code.co_filename,
                    "line": frame.f_lineno,
                    "function": frame.f_code.co_name,
                }
            )
            frame = frame.f_back
        self._cpu_samples.write(
            json.dumps(
                {
                    "elapsed_us": self.control.elapsed_us(),
                    "sampled_ns": now,
                    "native_thread_id": self.control.native_thread_id,
                    "stack": stack,
                }
            )
            + "\n"
        )
        self._cpu_samples.flush()

    def write(self, result: StepResult) -> None:
        """Write each presented frame with its aligned simulation observation."""
        desc = self._desc
        if desc is None or self._telemetry is None:
            raise RuntimeError("Debug capture is not open")
        if not self._paired_views:
            self._write_presentation(result)
            return
        if not isinstance(result, DebugFrameResult) or result.record is None:
            raise TypeError("Debug capture requires a DebugFrameResult")
        simulation_frame = (
            result.record.get("rollout_epoch", 0),
            result.record["frame"],
        )
        if simulation_frame <= self._last_simulation_frame:
            raise ValueError("Debug frames must arrive once, in simulation order")
        expected_shape = (1, 3, desc.video_height, 2 * desc.video_width)
        if (
            tuple(result.read_output().shape) != expected_shape
            or result.frame_count != 1
        ):
            raise ValueError(f"Debug frame pair must have shape {expected_shape}")
        rgb = (
            result_to_rgb24_tensor(
                result, desc, (2 * desc.video_width, desc.video_height)
            )
            .cpu()
            .numpy()
        )
        self._last_simulation_frame = simulation_frame
        physical = result.record["physics_vehicle"]
        self._physics_trail.append((physical["x_m"], physical["y_m"]))
        contact = (
            result.record["static_barrier_collision"]
            or result.record["actor_collision"]
        )
        if contact:
            self._contact_frames.append(self.frame_count)
        latest_frames = {}
        for index, (name, encoder) in enumerate(self._encoders.items()):
            if name == "physics":
                frame = render_physics_view(
                    result.record,
                    desc.video_width,
                    desc.video_height,
                    list(self._physics_trail),
                )[None]
            else:
                frame = np.ascontiguousarray(
                    rgb[:, :, index * desc.video_width : (index + 1) * desc.video_width]
                )
            encoder.write(frame)
            latest_frames[name] = frame[0].copy()
            if self.frame_count % self.snapshot_every == 0 or contact:
                Image.fromarray(frame[0]).save(
                    self.directory / "frames" / name / f"{self.frame_count:06d}.png"
                )
        self._telemetry.write(
            json.dumps(
                {
                    **result.record,
                    "capture_frame": self.frame_count,
                    "elapsed_us": self.control.elapsed_us(),
                },
                default=_json_value,
                allow_nan=False,
            )
            + "\n"
        )
        self._telemetry.flush()
        self._last_frames = latest_frames
        self.frame_count += 1

    def _write_presentation(self, result: StepResult) -> None:
        """Capture the ordinary HUD output and its actual presentation time."""
        assert self._desc is not None and self._telemetry is not None
        output = result.read_output()
        size = int(output.shape[-1]), int(output.shape[-2])
        if self._presentation_size is None:
            self._presentation_size = size
            self._encoders["presentation"] = Mp4Encoder(
                self.directory / "presentation.mp4",
                width=size[0],
                height=size[1],
                frames_per_second=self._desc.frames_per_second_for_ui,
            )
        if size != self._presentation_size:
            raise ValueError("Live capture presentation dimensions changed")
        rgb = result_to_rgb24_tensor(result, self._desc, size).cpu().numpy()
        self._encoders["presentation"].write(rgb)
        self._last_frames["presentation"] = rgb[-1].copy()
        now_ns = self._clock_ns()
        if self.frame_count % self.snapshot_every == 0:
            image = Image.fromarray(rgb[-1])
            image.save(
                self.directory
                / "frames"
                / "presentation"
                / f"{self.frame_count:06d}.png"
            )
            temporary = self.directory / "latest.tmp.png"
            image.save(temporary)
            temporary.replace(self.directory / "latest.png")
        self._telemetry.write(
            json.dumps(
                {
                    "frame": self.frame_count,
                    "ui_step": result.step_index,
                    "presented_ns": now_ns,
                    "session_time_s": (now_ns - self._started_ns) / 1e9,
                    "elapsed_us": self.control.elapsed_us(),
                }
            )
            + "\n"
        )
        self._telemetry.flush()
        self.frame_count += int(rgb.shape[0])

    def close(self) -> None:
        """Finalize videos and report any unconsumed commands."""
        try:
            with ExitStack() as cleanup:
                for encoder in self._encoders.values():
                    cleanup.callback(encoder.close)
                for name, frame in self._last_frames.items():
                    path = (
                        self.directory
                        / "frames"
                        / name
                        / f"{self.frame_count - 1:06d}.png"
                    )
                    if not path.exists():
                        Image.fromarray(frame).save(path)
            self._finalized = True
        finally:
            self.control.close()
            self._flush_control()
            for log in (self._model_steps, self._cpu_samples):
                if log is not None and not log.closed:
                    log.close()
            if self._control_socket is not None:
                self._control_socket.close()
                self._control_socket = None
                (self.directory / "control.sock").unlink(missing_ok=True)
            if self._input_log is not None:
                self._input_log.close()
                self._input_log = None
            if self._telemetry is not None:
                self._telemetry.close()
                self._telemetry = None
            if self._desc is not None:
                self._write_manifest()

    def _write_manifest(self) -> None:
        assert self._desc is not None
        metadata = self._desc.metadata[DEBUG_METADATA_KEY]
        manifest = {
            "format_version": 3,
            **{name: value for name, value in metadata.items() if name != "map_source"},
            "frames_written": self.frame_count,
            "finalized": self._finalized,
            "presentation_mode": self._desc.presentation_mode.value,
            "backpressure_mode": self._desc.backpressure_mode.value,
            "width": (
                self._presentation_size
                or (self._desc.video_width, self._desc.video_height)
            )[0],
            "height": (
                self._presentation_size
                or (self._desc.video_width, self._desc.video_height)
            )[1],
            "fps": self._desc.frames_per_second_for_ui
            if not self._paired_views
            else self._desc.frames_per_second_for_step,
            "snapshot_every": self.snapshot_every,
            "views": ["presentation"]
            if not self._paired_views
            else ["generated", "hdmap", "physics"],
            "contact_frames": self._contact_frames,
            "clock": "microseconds since gameplay ready (monotonic wall time)",
            "cpu_sample_ms": self._cpu_sample_ns / 1_000_000,
            "model_native_thread_id": self.control.native_thread_id,
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_value, allow_nan=False) + "\n",
            encoding="utf-8",
        )


class RobotaxiDebugMode(ClientWindowMode):
    """The ``robotaxi-debug`` output mode registered by the game package."""

    name = "robotaxi-debug"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--debug-output-dir",
            type=Path,
            help="Fresh directory for Robotaxi debug observations.",
        )
        parser.add_argument(
            "--debug-snapshot-every",
            type=int,
            default=30,
            help="Save all PNG views every N simulation frames (default: 30).",
        )

        parser.add_argument(
            "--debug-cpu-sample-ms",
            type=float,
            default=0,
            help="Sample model-thread Python stacks from the UI every N ms (0 disables).",
        )

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if parsed_args.debug_output_dir is None:
            raise ValueError("--debug-output-dir is required for robotaxi-debug")
        if parsed_args.debug_output_dir.expanduser().exists():
            raise ValueError(
                "--debug-output-dir must not already exist; use a fresh directory for each run"
            )
        if parsed_args.debug_snapshot_every <= 0:
            raise ValueError("--debug-snapshot-every must be positive")
        if (
            not math.isfinite(parsed_args.debug_cpu_sample_ms)
            or parsed_args.debug_cpu_sample_ms < 0
        ):
            raise ValueError("--debug-cpu-sample-ms must be finite and nonnegative")
        if shutil.which("ffmpeg") is None:
            raise ValueError("robotaxi-debug requires ffmpeg on PATH")

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        self.check_arguments(parsed_args)
        return RobotaxiDebugWindow(
            parsed_args.debug_output_dir.expanduser().resolve(),
            snapshot_every=parsed_args.debug_snapshot_every,
            cpu_sample_ms=parsed_args.debug_cpu_sample_ms,
        )

    def finished(self, client_window: IClientWindow) -> str:
        return str(cast(RobotaxiDebugWindow, client_window).directory / "manifest.json")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize debug value {type(value).__name__}")
