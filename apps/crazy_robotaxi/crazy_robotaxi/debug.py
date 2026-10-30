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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, TextIO, cast

import torch

from crazy_robotaxi.debug_capture import CaptureRecorder, _json_value
from crazy_robotaxi.debug_input import DEBUG_CONTROL_KEY, HeadlessInput
from crazy_robotaxi.ui import CrazyRobotaxiImGuiUILoop, TaxiHudState
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IUILoop, ModelInferenceState
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

DEBUG_METADATA_KEY = "robotaxi_debug"
"""Session metadata describing the selected map and driving script."""


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

    def step_ui(
        self, imgui: Any, step_index: int, events: UserInputEvents
    ) -> torch.Tensor | None:
        """Retain the source identity of the ordinary offscreen UI output."""
        control = self.state.debug_control
        if control is not None:
            control.presentation_feedback = None
        output = super().step_ui(imgui, step_index, events)
        if control is not None:
            frame = self.state._current
            control.presentation_source = (
                None if output is None or frame is None else frame.debug_record
            )
        return output

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
        capture_images: bool = True,
        image_buffer_chunks: int = 4,
        strict: bool = False,
        generation_delays_ms: tuple[float, ...] = (0.0,),
    ) -> None:
        if image_buffer_chunks <= 0:
            raise ValueError("image_buffer_chunks must be positive")
        self.image_buffer_chunks = image_buffer_chunks
        self.capture_images = capture_images
        self.strict = strict
        self.generation_delays_ms = generation_delays_ms
        self.recorder: CaptureRecorder | None = None
        self._input_status: dict[str, str] = {}
        self.directory = directory
        self.snapshot_every = snapshot_every
        self.frame_count = 0
        self._desc: SessionDesc | None = None
        self._input_events: list[UserInputEvent] = []
        self._clock_ns = clock_ns
        self._started_ns = 0
        self._paired_views = False
        self.control = HeadlessInput(clock_ns)
        self._model_steps: TextIO | None = None
        self._cpu_samples: TextIO | None = None
        self._cpu_sample_ns = int(cpu_sample_ms * 1_000_000)
        self._last_sample_ns = 0
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
        self.control.generation_delays_ms = self.generation_delays_ms
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
        self.recorder = CaptureRecorder(
            self.directory,
            session_desc,
            images=self.capture_images,
            snapshot_every=self.snapshot_every,
            image_slots=self.image_buffer_chunks,
        )
        self.control.capture = self.recorder
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
            if message.get("id") and message["status"] != "rejected":
                self._input_status[message["id"]] = message["status"]
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
        """Record presentation timing and its source independently of model capture."""
        if self.recorder is None:
            raise RuntimeError("Debug capture is not open")
        now_ns = self._clock_ns()
        source = (
            result.record
            if isinstance(result, DebugFrameResult)
            else self.control.presentation_source
        )
        identity = (
            None
            if source is None
            else {
                key: source[key]
                for key in (
                    "capture_frame",
                    "rollout_epoch",
                    "frame",
                    "model_step",
                    "autoregressive_index",
                    "chunk_frame",
                    "simulation_timestamp_us",
                )
            }
        )
        output = (
            None
            if self._paired_views or not self.capture_images
            else result.read_output()
        )
        if output is not None:
            self._presentation_size = int(output.shape[-1]), int(output.shape[-2])
        self.recorder.present(
            {
                "ui_step": result.step_index,
                "presented_ns": now_ns,
                "session_time_s": (now_ns - self._started_ns) / 1e9,
                "elapsed_us": self.control.elapsed_us(),
                "source": identity,
                "hud_feedback": self.control.presentation_feedback,
            },
            output,
        )
        self.frame_count += result.frame_count

    def close(self) -> None:
        """Drain the writer and report pending inputs and incomplete captures."""
        self.control.close()
        self._flush_control()
        if self.recorder is not None:
            self.recorder.close()
        for log in (self._model_steps, self._cpu_samples, self._input_log):
            if log is not None and not log.closed:
                log.close()
        if self._control_socket is not None:
            self._control_socket.close()
            self._control_socket = None
            (self.directory / "control.sock").unlink(missing_ok=True)
        if self._desc is not None:
            self._write_manifest()
            from crazy_robotaxi.debug_report import write_timing_report

            write_timing_report(self.directory)
        if (
            self.strict
            and self.recorder is not None
            and not self.recorder.summary()["capture_complete"]
        ):
            raise RuntimeError("Incomplete debug capture; see manifest.json")

    def _write_manifest(self) -> None:
        assert self._desc is not None
        metadata = self._desc.metadata[DEBUG_METADATA_KEY]
        manifest = {
            "format_version": 5,
            **{name: value for name, value in metadata.items() if name != "map_source"},
            "presentation_mode": self._desc.presentation_mode.value,
            "backpressure_mode": self._desc.backpressure_mode.value,
            "width": self._desc.video_width,
            "height": self._desc.video_height,
            "fps": self._desc.frames_per_second_for_step,
            "presentation_size": self._presentation_size,
            "presentation_fps": self._desc.frames_per_second_for_ui,
            "snapshot_every": self.snapshot_every,
            "image_buffer_chunks": self.image_buffer_chunks,
            "views": ["presentation"]
            if not self._paired_views
            else ["generated", "hdmap", "physics"],
            "clock": "microseconds since gameplay ready (monotonic wall time)",
            "cpu_sample_ms": self._cpu_sample_ns / 1_000_000,
            "debug_view": "paired" if self._paired_views else "gameplay",
            "generation_delays_ms": self.generation_delays_ms,
            "ready_ns": self.control.ready_ns,
            "model_native_thread_id": self.control.native_thread_id,
        }
        if self.recorder is not None:
            manifest.update(self.recorder.summary())
            pending = [
                identifier
                for identifier, status in self._input_status.items()
                if status not in {"consumed", "dispatched"}
            ]
            script = metadata.get("script_events", [])
            endpoints = [event for event in script if event["type"] == "close"]
            script_complete = (
                None
                if not script
                else (
                    bool(endpoints)
                    or self.recorder.termination_reason in {"block_limit", "game_ended"}
                )
                and all(
                    self._input_status.get(event["id"])
                    == ("dispatched" if event["type"] == "close" else "consumed")
                    for event in script
                )
            )
            manifest.update(
                pending_input_ids=pending,
                script_complete=script_complete,
                termination_reason=self.control.termination_reason
                or self.recorder.termination_reason
                or "runtime_stopped",
                complete=manifest["capture_complete"]
                and not pending
                and script_complete is not False
                and self.recorder.termination_reason != "model_error",
            )
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
            help="Save PNGs every N captured frames in each view (default: 30).",
        )

        parser.add_argument(
            "--debug-capture-images",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Save model/conditioning/physics images and UI video (default: enabled).",
        )
        parser.add_argument(
            "--debug-image-buffer-chunks",
            type=int,
            default=4,
            help="Maximum image chunks queued for encoding (default: 4). Increase to absorb capture bursts at the cost of GPU memory; sustained encoder overload still drops frames.",
        )
        parser.add_argument(
            "--debug-strict-capture",
            action="store_true",
            help="Fail the run if any capture data is missing or could not be written.",
        )
        parser.add_argument(
            "--debug-cpu-sample-ms",
            type=float,
            default=0,
            help="Sample model-thread Python stacks from the UI every N ms (0 disables).",
        )
        parser.add_argument(
            "--debug-generation-delay-ms",
            type=float,
            nargs="+",
            default=[0.0],
            help="Additional wait before generation, in ms. One value is constant; multiple values repeat per chunk after warmup. Does not simulate CPU/GPU contention.",
        )

    def check_arguments(self, parsed_args: argparse.Namespace) -> None:
        if parsed_args.debug_output_dir is None:
            raise ValueError("--debug-output-dir is required for robotaxi-debug")
        if parsed_args.debug_output_dir.expanduser().exists():
            raise ValueError(
                "--debug-output-dir must not already exist; use a fresh directory for each run"
            )
        if parsed_args.debug_image_buffer_chunks <= 0:
            raise ValueError("--debug-image-buffer-chunks must be positive")
        if parsed_args.debug_snapshot_every <= 0:
            raise ValueError("--debug-snapshot-every must be positive")
        if (
            not math.isfinite(parsed_args.debug_cpu_sample_ms)
            or parsed_args.debug_cpu_sample_ms < 0
        ):
            raise ValueError("--debug-cpu-sample-ms must be finite and nonnegative")
        if parsed_args.debug_capture_images and shutil.which("ffmpeg") is None:
            raise ValueError("robotaxi-debug requires ffmpeg on PATH")
        if any(
            not math.isfinite(value) or value < 0
            for value in parsed_args.debug_generation_delay_ms
        ):
            raise ValueError(
                "--debug-generation-delay-ms must be finite and nonnegative"
            )

    def create(self, parsed_args: argparse.Namespace) -> IClientWindow:
        self.check_arguments(parsed_args)
        return RobotaxiDebugWindow(
            parsed_args.debug_output_dir.expanduser().resolve(),
            snapshot_every=parsed_args.debug_snapshot_every,
            cpu_sample_ms=parsed_args.debug_cpu_sample_ms,
            capture_images=parsed_args.debug_capture_images,
            image_buffer_chunks=parsed_args.debug_image_buffer_chunks,
            strict=parsed_args.debug_strict_capture,
            generation_delays_ms=tuple(parsed_args.debug_generation_delay_ms),
        )

    def finished(self, client_window: IClientWindow) -> str:
        return str(cast(RobotaxiDebugWindow, client_window).directory / "manifest.json")
