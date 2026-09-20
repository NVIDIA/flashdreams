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

"""Default UI loop for presenting model output."""

from typing import Generic, TypeVar, final

from torch import Tensor

from flashdreams.api_v2.loop import IUILoop, ModelInferenceState
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_StateT = TypeVar("_StateT")


class BlitModelOutputToScreenLoop(IUILoop[_StateT], Generic[_StateT]):
    """Draw every model channel into one UI frame."""

    def _initialize_loop_state(self) -> None:
        self._last_presented_frame_count = 0

    def frames_to_blit(self) -> tuple[Tensor, ...]:
        """Return the model channels to draw, bottom channel first.

        Subclasses that present a subset (a headless run that wants the
        generated video without the debug channels, say) override this.
        """
        return self.presented_model_frames()

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult | None:
        """Draw the model channels in list order."""
        del events
        output = None
        for frame in self.frames_to_blit():
            output = self._presentation_manager.composite(output, frame)
        self._last_presented_frame_count = (
            self._presentation_manager.presented_frame_count
        )
        if output is None:
            return None
        return StepResult(
            step_index=step_index,
            output=_frame_to_layout(output, self.output_layout),
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
        self._last_presented_frame_count = 0


def _frame_to_layout(frame: Tensor, layout: VideoTensorLayout) -> Tensor:
    """Add singleton time, batch, and view dimensions for ``layout``."""
    if layout is VideoTensorLayout.tchw:
        return frame.unsqueeze(0)
    if layout is VideoTensorLayout.btchw:
        return frame.unsqueeze(0).unsqueeze(0)
    if layout is VideoTensorLayout.bcthw:
        return frame.unsqueeze(0).unsqueeze(2)
    if layout is VideoTensorLayout.bvtchw:
        return frame.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported presentation layout: {layout}.")


__all__ = ["BlitModelOutputToScreenLoop"]
