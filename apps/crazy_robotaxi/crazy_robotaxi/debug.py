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
from pathlib import Path
from typing import Any, cast

import torch
from omnidreams_game_engine.debug_window import DebugCaptureWindow, DebugFrameResult

from crazy_robotaxi.ui import CrazyRobotaxiImGuiUILoop, TaxiHudState
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IUILoop, ModelInferenceState
from flashdreams.runtime_v2.client_window_factory import ClientWindowMode
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

DEBUG_METADATA_KEY = "robotaxi_debug"
"""Session metadata describing the selected map and driving script."""


class RobotaxiDebugUILoop(IUILoop[TaxiHudState]):
    """Present every generated/HD-map pair without a graphics window or HUD."""

    def _initialize_loop_state(self) -> None:
        self._last_presented_count = 0

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Package both model channels and their matching telemetry for the sink."""
        self.state.consume_input_events(events)
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


class RobotaxiDebugWindow(DebugCaptureWindow):
    """Capture the shared runtime with Robotaxi inputs and game outcome reports."""

    metadata_key = DEBUG_METADATA_KEY

    def _save_inputs(self, metadata):
        super()._save_inputs(metadata)
        if "map_source" in metadata:
            inputs = self.directory / "inputs"
            inputs.mkdir(exist_ok=True)
            (inputs / Path(metadata["map"]).name).write_text(
                metadata["map_source"], encoding="utf-8"
            )
            (inputs / "settings.yaml").write_text(
                json.dumps(metadata["settings"], indent=2) + "\n", encoding="utf-8"
            )

    def _manifest_metadata(self, metadata):
        return {name: value for name, value in metadata.items() if name != "map_source"}

    def _write_report(self):
        from crazy_robotaxi.debug_report import write_timing_report

        write_timing_report(self.directory)


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
            "--debug-record-buffer-events",
            type=int,
            default=256,
            help="Maximum queued telemetry, presentation, and image jobs (default: 256). Increase alongside the image budget when encoding falls behind; overflow remains a capture failure.",
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
        if parsed_args.debug_record_buffer_events <= 0:
            raise ValueError("--debug-record-buffer-events must be positive")
        if parsed_args.debug_snapshot_every <= 0:
            raise ValueError("--debug-snapshot-every must be positive")
        if (
            not math.isfinite(parsed_args.debug_cpu_sample_ms)
            or parsed_args.debug_cpu_sample_ms < 0
        ):
            raise ValueError("--debug-cpu-sample-ms must be finite and nonnegative")
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
            record_buffer_events=parsed_args.debug_record_buffer_events,
            strict=parsed_args.debug_strict_capture,
            generation_delays_ms=tuple(parsed_args.debug_generation_delay_ms),
        )

    def finished(self, client_window: IClientWindow) -> str:
        return str(cast(RobotaxiDebugWindow, client_window).directory / "manifest.json")
