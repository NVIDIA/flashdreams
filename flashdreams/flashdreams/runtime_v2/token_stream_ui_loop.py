# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UI loop for the token-streaming path: forward whole latent chunks."""

from typing import Generic, TypeVar, final

from flashdreams.api_v2.loop import IUILoop, ModelInferenceState
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_StateT = TypeVar("_StateT")


class TokenStreamUILoop(IUILoop[_StateT], Generic[_StateT]):
    """Forward each presented latent chunk to the client, uncomposited.

    The presentation manager runs in latent passthrough, so a whole latent
    chunk is selected per tick with no fps pacing. This loop returns that chunk
    as-is for the window (the WebRTC server) to stream; the browser decodes it.
    No RGB compositing happens here. Each chunk is emitted exactly once,
    deduplicated by ``(generation, step_index)`` because a CONTINUOUS
    presentation re-runs :meth:`step` every tick while the same chunk is held.
    """

    def _initialize_loop_state(self) -> None:
        self._last_emitted: tuple[int, int] | None = None

    @final
    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Return the newly presented latent chunk, or nothing."""
        del step_index, events
        result = self.presented_latent_result()
        if result is None:
            return []
        key = (self._generation, result.step_index)
        if key == self._last_emitted:
            # Same chunk as last tick; the client already received it.
            return []
        self._last_emitted = key
        return [result]

    def is_finished(self) -> bool:
        return (
            self.model_inference_state is ModelInferenceState.FINISHED
            and not self.has_pending_model_frames()
        )

    def reset(self) -> None:
        self._last_emitted = None


__all__ = ["TokenStreamUILoop"]
