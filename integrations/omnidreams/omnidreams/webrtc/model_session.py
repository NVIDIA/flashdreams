# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous OmniDreams conditioning session used by WebRTC serving."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from omnidreams.conditioning.conditioning_wrapper import (
    OmnidreamsConditioningState,
    OmnidreamsConditioningWrapper,
    TextPrompt,
)

from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import StepResult

OutputStreamFactory = Callable[[], VideoOutputStream]


class OmnidreamsConditioningSessionCore:
    """Own one conditioning-wrapper rollout and its generated output stream."""

    def __init__(
        self,
        *,
        wrapper: OmnidreamsConditioningWrapper,
        output_stream_factory: OutputStreamFactory,
    ) -> None:
        self.wrapper = wrapper
        self._output_stream_factory = output_stream_factory
        self._output_stream = output_stream_factory()
        self._state: OmnidreamsConditioningState | None = None
        self._step_index = 0
        self._renderer: Any | None = None
        self._text_prompts: list[TextPrompt] | None = None
        self._initial_rgb_frames: torch.Tensor | None = None
        self._closed = False

    @property
    def step_index(self) -> int:
        return self._step_index

    def next_num_frames(self) -> int:
        self._require_open()
        if self._state is None:
            return int(self.wrapper.initial_frame_chunk_size)
        return int(self.wrapper.frame_chunk_size)

    def reset(
        self,
        *,
        renderer: Any,
        text_prompts: list[TextPrompt],
        initial_rgb_frames: torch.Tensor,
    ) -> None:
        self._require_open()
        self._discard_state(cleanup_renderer=False)
        self._renderer = renderer
        self._text_prompts = text_prompts
        self._initial_rgb_frames = initial_rgb_frames
        self._step_index = 0

    def step(
        self,
        *,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        serve_hdmaps: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepResult:
        self._require_initialized()
        renderer = self._renderer
        text_prompts = self._text_prompts
        initial_rgb_frames = self._initial_rgb_frames
        if renderer is None or text_prompts is None or initial_rgb_frames is None:
            raise RuntimeError("OmniDreams conditioning session is not initialized.")
        expected_frames = self.next_num_frames()
        if self._state is None:
            output = self.wrapper.start_generation(
                text_prompts=text_prompts,
                initial_rgb_frames=initial_rgb_frames,
                renderer=renderer,
                camera_names=camera_names,
                camera_poses_per_view=camera_poses_per_view,
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
        else:
            output = self.wrapper.continue_generation(
                state=self._state,
                camera_names=camera_names,
                camera_poses_per_view=camera_poses_per_view,
                frame_timestamps_us=frame_timestamps_us,
                skip_video_generation=serve_hdmaps,
            )
        self._state = output.state
        if self._state.pipeline_cache is not None:
            self.wrapper.finalize_block_generation(
                self._state.pipeline_cache,
                output.finalization_state,
            )

        if serve_hdmaps:
            result = StepResult.from_video_chunk(
                step_index=self._step_index,
                video_chunk=output.condition_frames.detach(),
                layout="bvtchw",
                metadata=metadata,
            )
        else:
            if output.rgb_frames is None:
                raise RuntimeError("OmniDreams conditioning produced no RGB frames.")
            result = self._output_stream.process(
                output.rgb_frames,
                autoregressive_index=self._step_index,
                metadata=metadata,
            )
        if result.frame_count != expected_frames:
            raise RuntimeError(
                f"Expected generated chunk to contain {expected_frames} frames, "
                f"got {result.frame_count}."
            )
        self._step_index += 1
        return result

    def replace_output_stream(self, output_stream_factory: OutputStreamFactory) -> None:
        self._require_open()
        self._output_stream.finish()
        self._output_stream_factory = output_stream_factory
        self._output_stream = output_stream_factory()

    @property
    def postprocess_stream(self) -> object | None:
        return self._output_stream.postprocess_stream

    def close(self) -> None:
        if self._closed:
            return
        self._discard_state(cleanup_renderer=True)
        self._output_stream.finish()
        self._closed = True

    def _discard_state(self, *, cleanup_renderer: bool) -> None:
        if self._state is not None and cleanup_renderer:
            self.wrapper.cleanup(self._state)
        elif self._state is not None and self._state.pipeline_cache is not None:
            del self._state.pipeline_cache
        self._state = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OmniDreams conditioning session is closed.")

    def _require_initialized(self) -> None:
        self._require_open()
        if (
            self._renderer is None
            or self._text_prompts is None
            or self._initial_rgb_frames is None
        ):
            raise RuntimeError("OmniDreams conditioning session is not initialized.")


__all__ = ["OmnidreamsConditioningSessionCore"]
