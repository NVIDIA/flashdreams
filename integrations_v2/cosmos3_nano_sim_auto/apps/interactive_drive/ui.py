# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live driving controls and camera streams for the shared sensor viewer."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from interactive_drive.app import InteractiveDriveSession, InteractiveDriveUIState
from interactive_drive.config import PresentationLayout
from interactive_drive.core import (
    InteractiveDriveModelLoop,
    InteractiveDriveModelState,
    _frame_chunk_tensor,
    _output_cuda_device,
)
from interactive_drive.multiview import AuxiliaryView, MultiviewUILoop, MultiviewUIState
from interactive_drive.types import FrameChunk
from torch import Tensor

from cosmos3_nano_sim_auto.impl.ui import camera_views_for_names
from flashdreams.api_v2.loop import IModelLoop, invoke_async
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class MultiviewDriveModelLoop(InteractiveDriveModelLoop):
    """Present each camera and optional LiDAR stream in its own channel."""

    def _frame_chunk_tensors(self, chunk: FrameChunk) -> list[Tensor]:
        state = self.state
        if state.config.no_ui:
            return super()._frame_chunk_tensors(chunk)
        if not chunk.frames:
            return []
        return [
            _frame_chunk_tensor(
                chunk,
                state.view_mode,
                layout=PresentationLayout(1, 1, view),
                output_device=_output_cuda_device(state.config.app),
            )
            for view in range(chunk.frames[0].views)
        ]


class MultiviewDriveUILoop(MultiviewUILoop):
    """Keep live controls active while the viewer arranges or focuses sensors."""

    def _apply_inputs(self, events: UserInputEvents) -> None:
        self.state.drive_input.apply(events)

    def _draw_view_controls(self, imgui: Any) -> None:
        state = self.state
        self._draw_scene_controls(imgui)
        if imgui.button(f"View: {state.view_mode.upper()}"):
            self._toggle_view()
        if state.show_postprocess_toggle:
            enabled = (
                state.telemetry.postprocess_enabled
                if state.telemetry is not None
                else state.postprocess_enabled
            )
            changed, enabled = imgui.checkbox("Post-processing", enabled)
            if changed:
                state.postprocess_enabled = bool(enabled)
                invoke_async(
                    state.model_loop,
                    lambda model_state, enabled=bool(enabled): (
                        model_state.set_postprocess_enabled(enabled)
                    ),
                )

    def _draw_drive_hud(self, imgui: Any) -> None:
        state = self.state
        command = state.drive_input.command()
        telemetry = state.telemetry
        if telemetry is None:
            imgui.text_wrapped("Waiting for the first driving block...")
        else:
            imgui.text(f"Block {telemetry.blocks_generated}")
            imgui.text(f"Speed  {abs(telemetry.speed_mps) * 2.236936:5.1f} mph")
            imgui.text(f"Gear   {'R' if telemetry.reverse else 'D'}")
            imgui.text(f"model_loop_ms: {telemetry.model_loop_ms:.1f}")
            if telemetry.warning:
                imgui.push_style_color(
                    imgui.Col_.text, imgui.ImVec4(1.0, 0.25, 0.25, 1.0)
                )
                try:
                    imgui.text_wrapped("MAP COVERAGE WARNING")
                    imgui.text_wrapped(telemetry.warning)
                finally:
                    imgui.pop_style_color()
        imgui.text(f"Input  {state.drive_input.source()}")
        imgui.text(f"Steer  {command.steer:+.2f}")
        self._progress(imgui, "Throttle", command.throttle)
        self._progress(imgui, "Brake", command.brake)

    def _draw_help(self, imgui: Any) -> None:
        imgui.text_wrapped(
            "W drives forward; S reverses; Space brakes; A/D steer. "
            "R restarts the rollout. Gamepads and wheels are supported. "
            "1: RGB, 2: HD map."
        )


class MultiviewDriveSession(InteractiveDriveSession):
    """Bind the live model's sensor channels to the bearing-aware viewer."""

    model_loop_type = MultiviewDriveModelLoop
    ui_loop_type = MultiviewDriveUILoop
    auxiliary_views: tuple[AuxiliaryView, ...] = ()
    """Optional sensor metadata following the camera channels."""

    def _create_ui_state(
        self, model_loop: IModelLoop[InteractiveDriveModelState]
    ) -> MultiviewUIState:
        base = super()._create_ui_state(model_loop)
        selected = self._config.app.presentation.view_index
        return MultiviewUIState(
            **{
                item.name: getattr(base, item.name)
                for item in fields(InteractiveDriveUIState)
            },
            cameras=camera_views_for_names(self._config.app.camera_names),
            auxiliary_views=self.auxiliary_views,
            focus_channel=-1 if selected is None else selected,
        )
