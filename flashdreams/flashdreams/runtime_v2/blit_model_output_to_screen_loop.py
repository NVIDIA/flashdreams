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

from typing import final

import torch
from torch import Tensor

from flashdreams.api_v2.loop import IUILoop
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


class BlitModelOutputToScreenLoop(IUILoop[None]):
    """Draw every model channel into one UI frame."""

    @final
    def should_redraw_for_input(self, events: UserInputEvents) -> bool:
        """Ignore input because the default blitter has no interactive state."""
        del events
        return False

    @final
    def step(self, step_index: int, events: UserInputEvents) -> StepResult | None:
        """Draw the model channels in list order."""
        del events
        output = None
        for frame in self.presented_model_frames():
            output = self._presentation_manager.composite(output, frame)
        if output is None:
            return None
        output_ready_event = None
        if output.is_cuda:
            output_ready_event = torch.cuda.Event()
            output_ready_event.record(torch.cuda.current_stream(output.device))
        return StepResult(
            step_index=step_index,
            output=_frame_to_layout(output, self.output_layout),
            frame_count=1,
            output_layout=self.output_layout,
            output_ready_event=output_ready_event,
        )

    def reset(self) -> None:
        return


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
