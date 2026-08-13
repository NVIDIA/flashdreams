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

"""Text-to-video model runtime and one-time pipeline state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.runtime import InferenceInput
from flashdreams_runner import AppConfig, IOHandler, Runtime, Session
from flashdreams_runner.webrtc import WebRTCMode

from .session import T2VScenario, T2VSession, T2VSessionDefaults


@dataclass(frozen=True, slots=True)
class T2VArtifact:
    """Completed WebRTC recording and the scenario that produced it."""

    path: Path
    """Path to the generated MP4 file."""

    scenario: T2VScenario
    """Prompt, duration, and video geometry stored with the recording."""


class T2VRuntime(Runtime):
    """Own T2V model weights and create isolated generation sessions."""

    def __init__(
        self,
        *,
        pipeline_config: StreamInferencePipelineConfig,
        session_defaults: T2VSessionDefaults,
        config: AppConfig,
    ) -> None:
        self._pipeline_config = pipeline_config
        self._session_defaults = session_defaults
        self._config = config
        self._pipeline: StreamInferencePipeline[Any, Any, Any] | None = None
        self._io_handler: IOHandler | None = None
        self._record_sessions = False
        self._latest_artifact: T2VArtifact | None = None

    @property
    def config(self) -> AppConfig:
        """Return T2V configuration for runner-owned presentation."""
        return self._config

    def initialize(self, *, device: str, io_handler: IOHandler) -> None:
        """Construct model weights once for the selected device and I/O mode."""
        if self._pipeline is not None:
            raise RuntimeError("T2VRuntime is already initialized.")
        pipeline = self._pipeline_config.setup()
        if not isinstance(pipeline, StreamInferencePipeline):
            raise TypeError(
                "T2V pipeline config must construct StreamInferencePipeline, got "
                f"{type(pipeline).__name__}."
            )
        self._pipeline = pipeline.to(device).eval()
        self._io_handler = io_handler
        if isinstance(io_handler, WebRTCMode):
            from .webrtc import T2VWebRTCCustomization

            self._record_sessions = True
            io_handler.customize(T2VWebRTCCustomization(runtime=self))

    def create_session(self, initial_input: InferenceInput | None = None) -> Session:
        """Create a T2V session with its own prompt and autoregressive cache."""
        if self._pipeline is None:
            raise RuntimeError("T2VRuntime must be initialized before use.")
        return T2VSession(
            pipeline=self._pipeline,
            defaults=self._session_defaults,
            initial_input=initial_input or InferenceInput(),
            output_layout=self._config.output_layout,
            record_artifact=self._record_artifact if self._record_sessions else None,
        )

    def prepare_session_input(
        self,
        *,
        prompt: str | None = None,
        total_blocks: int | None = None,
    ) -> InferenceInput:
        """Build complete initial input for a browser-created T2V session."""
        return InferenceInput(
            global_conditioning={
                "prompt": self._session_defaults.prompt if prompt is None else prompt,
                "total_blocks": (
                    self._session_defaults.total_blocks
                    if total_blocks is None
                    else total_blocks
                ),
                "pixel_height": self._session_defaults.pixel_height,
                "pixel_width": self._session_defaults.pixel_width,
                "fps": self._session_defaults.fps,
            }
        )

    def blocks_for_duration(self, duration_s: float) -> int:
        """Return enough autoregressive blocks for a requested duration."""
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be finite and > 0.")
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("T2VRuntime must be initialized before use.")
        target_frames = math.ceil(duration_s * self._session_defaults.fps)
        generated_frames = 0
        block_index = 0
        pipeline_api = cast(Any, pipeline)
        while generated_frames < target_frames:
            block_frames = int(pipeline_api.get_num_output_frames(block_index))
            if block_frames <= 0:
                raise ValueError("T2V pipeline output frame counts must be > 0.")
            generated_frames += block_frames
            block_index += 1
        return block_index

    def peek_steady_output_num_frames(self) -> int:
        """Return the steady chunk size used to bound WebRTC delivery queues."""
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("T2VRuntime must be initialized before use.")
        return int(cast(Any, pipeline).get_num_output_frames(1))

    @property
    def latest_artifact(self) -> T2VArtifact | None:
        """Return the most recently completed WebRTC recording."""
        return self._latest_artifact

    def _record_artifact(self, path: Path, scenario: T2VScenario) -> None:
        self._latest_artifact = T2VArtifact(path=path, scenario=scenario)

    def destroy(self) -> None:
        """Release pipeline weights and accelerator allocator state."""
        pipeline = self._pipeline
        self._pipeline = None
        self._io_handler = None
        self._record_sessions = False
        if pipeline is None:
            return
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["T2VArtifact", "T2VRuntime"]
