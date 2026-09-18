# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HUD-free UI loop that presents raw model frames for headless runs."""

from __future__ import annotations

from typing import final

from flashdreams.api_v2.loop import IUILoop, ModelInferenceState
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import frame_to_layout
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from crazy_robotaxi.ui import TaxiHudState


class CrazyRobotaxiHeadlessUILoop(IUILoop[TaxiHudState]):
    """Present the generated video channel without drawing the ImGui HUD.

    The HUD state still receives every ``invoke_async`` message from the model
    thread (menu selection, loading status, published frames), so the startup
    flow driven by ``--game-mode`` and ``--map`` runs unchanged. Only the video
    channel is presented; the HD-map and BEV channels are dropped. Nothing here
    needs a Vulkan device, which makes ``--mode mp4`` runs possible on hosts
    without a SlangPy-compatible graphics adapter (for example WSL2).
    """

    def _initialize_loop_state(self) -> None:
        self._last_presented_frame_count = 0

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult | None:
        """Return the presented video frame, or nothing before the first one."""
        del events
        frames = self.presented_model_frames()
        self._last_presented_frame_count = (
            self._presentation_manager.presented_frame_count
        )
        if not frames:
            return None
        output = self._presentation_manager.composite(None, frames[0])
        return StepResult(
            step_index=step_index,
            output=frame_to_layout(output, self.output_layout),
            frame_count=1,
            output_layout=self.output_layout,
        )

    def is_finished(self) -> bool:
        return (
            self.model_inference_state is ModelInferenceState.FINISHED
            and not self._presentation_manager.has_pending_frames()
            and self._last_presented_frame_count
            == self._presentation_manager.presented_frame_count
        )

    def reset(self) -> None:
        self.state.reset()
        self._last_presented_frame_count = 0


__all__ = ["CrazyRobotaxiHeadlessUILoop"]
