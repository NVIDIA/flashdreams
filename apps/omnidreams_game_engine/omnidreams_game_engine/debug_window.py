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

import json
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, TextIO

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.user_input_event import UserInputEvent
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from omnidreams_game_engine.debug_capture import CaptureRecorder, _json_value
from omnidreams_game_engine.debug_input import DEBUG_CONTROL_KEY, HeadlessInput

DEBUG_METADATA_KEY = "omnidreams_debug"


@dataclass(frozen=True, slots=True, match_args=False)
class DebugFrameResult(StepResult):
    """A generated/HD-map frame pair with its aligned simulation record."""

    record: dict[str, Any] | None = None


class DebugCaptureWindow(IClientWindow):
    """Write synchronized views, contact PNGs, and aligned JSONL observations."""

    metadata_key = DEBUG_METADATA_KEY

    def __init__(
        self,
        directory: Path,
        *,
        snapshot_every: int = 30,
        cpu_sample_ms: float = 0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        capture_images: bool = True,
        image_buffer_chunks: int = 4,
        record_buffer_events: int = 256,
        strict: bool = False,
        generation_delays_ms: tuple[float, ...] = (0.0,),
    ) -> None:
        if image_buffer_chunks <= 0:
            raise ValueError("image_buffer_chunks must be positive")
        if record_buffer_events <= 0:
            raise ValueError("record_buffer_events must be positive")
        self.image_buffer_chunks = image_buffer_chunks
        self.record_buffer_events = record_buffer_events
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
        if self.metadata_key not in session_desc.metadata:
            raise ValueError("Debug capture requires session capture metadata")
        metadata = session_desc.metadata[self.metadata_key]
        self._paired_views = metadata.get("paired_views", False)
        self.control = session_desc.metadata.get(DEBUG_CONTROL_KEY, self.control)
        self.control.generation_delays_ms = self.generation_delays_ms
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError("Debug capture requires tchw output")
        if session_desc.video_width % 2 or session_desc.video_height % 2:
            raise ValueError("Debug videos require even pixel dimensions")
        self.directory.mkdir(parents=True, exist_ok=False)
        self._desc = session_desc
        metadata = session_desc.metadata[self.metadata_key]
        self._save_inputs(metadata)
        self.recorder = CaptureRecorder(
            self.directory,
            session_desc,
            images=self.capture_images,
            snapshot_every=self.snapshot_every,
            image_slots=self.image_buffer_chunks,
            queue_size=self.record_buffer_events,
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
            if message.get("id") and message["status"] in {
                "accepted",
                "dispatched",
                "consumed",
                "cancelled",
            }:
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
            self._write_report()
        if (
            self.strict
            and self.recorder is not None
            and not self.recorder.summary()["capture_complete"]
        ):
            raise RuntimeError("Incomplete debug capture; see manifest.json")

    def _write_manifest(self) -> None:
        assert self._desc is not None
        metadata = self._desc.metadata[self.metadata_key]
        manifest = {
            "format_version": 5,
            **self._manifest_metadata(metadata),
            "presentation_mode": self._desc.presentation_mode.value,
            "backpressure_mode": self._desc.backpressure_mode.value,
            "width": self._desc.video_width,
            "height": self._desc.video_height,
            "fps": self._desc.frames_per_second_for_step,
            "presentation_size": self._presentation_size,
            "presentation_fps": self._desc.frames_per_second_for_ui,
            "snapshot_every": self.snapshot_every,
            "image_buffer_chunks": self.image_buffer_chunks,
            "record_buffer_events": self.record_buffer_events,
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

    def _save_inputs(self, metadata: dict[str, Any]) -> None:
        if metadata.get("script_events"):
            inputs = self.directory / "inputs"
            inputs.mkdir(exist_ok=True)
            (inputs / "drive.yaml").write_text(
                json.dumps({"events": metadata["script_events"]}, indent=2) + "\n",
                encoding="utf-8",
            )

    def _manifest_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return metadata

    def _write_report(self) -> None:
        from omnidreams_game_engine.debug_report import write_timing_report

        write_timing_report(self.directory)
