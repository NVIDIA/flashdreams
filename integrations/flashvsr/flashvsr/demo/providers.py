# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""New-API model input provider for decoded FlashVSR videos."""

from __future__ import annotations

import torch

from flashdreams.runtime import InferenceInput, InferenceInputSchema
from flashdreams.runtime.demo import (
    ControlDecision,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.types import StepRequirements
from flashvsr.runtime import (
    FIELD_VALID_FRAME_COUNT,
    FIELD_VIDEO_CHUNK,
)

from .spec import PreparedFlashVSRVideo

PREPARED_VIDEO_METADATA_KEY = "prepared_video"


class FlashVSRVideoInputProvider:
    """Slice one decoded video into exact model-requested frame chunks."""

    def __init__(
        self,
        *,
        scenario: PreparedScenario,
        inference_input_schema: InferenceInputSchema,
    ) -> None:
        prepared = scenario.metadata.get(PREPARED_VIDEO_METADATA_KEY)
        if not isinstance(prepared, PreparedFlashVSRVideo):
            raise TypeError(
                "FlashVSR prepared scenario is missing its decoded video source."
            )
        self.prepared = prepared
        self._initial_input = scenario.initial_inputs
        self._cursor = 0
        self._closed = False
        self.capabilities = ProviderCapabilities(
            supports_realtime_clock=True,
            supports_recorded_input=True,
            supports_reset=True,
            deterministic_given_inputs=True,
            user_input_schema=scenario.source_schema,
            inference_input_schema=inference_input_schema,
        )

    def prepare_initial_input(self) -> InferenceInput:
        self._require_open()
        return self._initial_input

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del user_window
        self._require_open()
        requested = request.input_frame_count
        expected_valid = int(request.metadata.get(FIELD_VALID_FRAME_COUNT, requested))
        source_start = self._cursor
        if self.prepared.scenario.loop_input:
            chunk = self._looping_chunk(requested)
            valid = requested
        else:
            chunk, valid = self._finite_chunk(requested)
        if valid <= 0:
            return PreparedStep(
                control=ControlDecision(
                    close_session=True,
                    reason="FlashVSR source video is exhausted.",
                )
            )
        if valid != expected_valid:
            raise RuntimeError(
                "FlashVSR provider/session frame-count disagreement: "
                f"session requested {expected_valid} valid frames, provider has {valid}."
            )
        return PreparedStep(
            inference_input=InferenceInput(
                step={FIELD_VIDEO_CHUNK: chunk},
                metadata={
                    FIELD_VALID_FRAME_COUNT: valid,
                    "source_frame_start": source_start,
                },
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._require_open()
        self._cursor = 0

    def close(self) -> None:
        self._closed = True

    def _finite_chunk(self, requested: int) -> tuple[torch.Tensor, int]:
        video = self.prepared.video
        total = self.prepared.total_frames
        valid = min(requested, total - self._cursor)
        if valid <= 0:
            return video[:, :, :0], 0
        chunk = video[:, :, self._cursor : self._cursor + valid]
        self._cursor += valid
        if valid == requested:
            return chunk, valid
        if self.prepared.scenario.tail_policy != "pad":
            return chunk, valid
        padding = chunk[:, :, -1:].expand(-1, -1, requested - valid, -1, -1)
        return torch.cat((chunk, padding), dim=2), valid

    def _looping_chunk(self, requested: int) -> torch.Tensor:
        video = self.prepared.video
        total = self.prepared.total_frames
        indices = (torch.arange(requested) + self._cursor) % total
        chunk = video.index_select(2, indices)
        self._cursor = (self._cursor + requested) % total
        return chunk

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("FlashVSR video input provider is closed.")


__all__ = [
    "PREPARED_VIDEO_METADATA_KEY",
    "FlashVSRVideoInputProvider",
]
