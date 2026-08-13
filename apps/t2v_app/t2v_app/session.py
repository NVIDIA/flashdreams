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

"""Text-to-video session state and generation loop iteration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import torch

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime import InferenceInput, StepRequest, StepResult
from flashdreams.runtime.video_output import Mp4VideoOutputTarget
from flashdreams_runner import Session

FIELD_PROMPT = "prompt"
FIELD_TOTAL_BLOCKS = "total_blocks"
FIELD_PIXEL_HEIGHT = "pixel_height"
FIELD_PIXEL_WIDTH = "pixel_width"
FIELD_FPS = "fps"


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VScenario:
    """Validated state used by one text-to-video session."""

    prompt: str
    """Text prompt used to initialize the model cache."""

    total_blocks: int | None
    """Generation limit, or ``None`` for an externally driven session."""

    pixel_height: int
    """Output height in pixels."""

    pixel_width: int
    """Output width in pixels."""

    fps: int
    """Output video frame rate."""


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VSessionDefaults:
    """Default state copied into each new T2V session."""

    prompt: str
    """Text prompt used when the session input does not override it."""

    total_blocks: int | None
    """Optional session-owned generation limit."""

    pixel_height: int
    """Output height used when the session input does not override it."""

    pixel_width: int
    """Output width used when the session input does not override it."""

    fps: int
    """Output frame rate used for WebRTC recordings."""


class T2VSession(Session):
    """Own one prompt, autoregressive cache, and T2V generation loop."""

    def __init__(
        self,
        *,
        pipeline: StreamInferencePipeline[Any, Any, Any],
        defaults: T2VSessionDefaults,
        initial_input: InferenceInput,
        output_layout: VideoTensorLayout,
        record_artifact: Callable[[Path, T2VScenario], None] | None = None,
        recording_directory: Path | None = None,
    ) -> None:
        scenario = _session_scenario(defaults, initial_input)
        decoder = pipeline.decoder
        if not isinstance(decoder, StreamingVideoDecoder):
            raise TypeError("T2V pipelines require a StreamingVideoDecoder.")
        ratio = decoder.spatial_compression_ratio
        pixel_height = scenario.pixel_height
        pixel_width = scenario.pixel_width
        if pixel_height % ratio or pixel_width % ratio:
            raise ValueError(
                "T2V dimensions must be divisible by the decoder spatial "
                f"compression ratio ({ratio})."
            )

        self._pipeline = pipeline
        self._scenario = scenario
        self._prompt = scenario.prompt
        self._pixel_height = pixel_height
        self._pixel_width = pixel_width
        self._output_layout: VideoTensorLayout = output_layout
        pipeline_api = cast(Any, pipeline)
        self._cache: object | None = pipeline_api.initialize_cache(
            text=[self._prompt],
            image=None,
            height=pixel_height // ratio,
            width=pixel_width // ratio,
        )
        self._step_index = 0
        self._steady_output_frame_count = int(pipeline_api.get_num_output_frames(1))
        self._destroyed = False
        self._record_artifact = record_artifact
        self._artifact_path: Path | None = None
        self._artifact_output: Mp4VideoOutputTarget | None = None
        if record_artifact is not None:
            recording_directory = recording_directory or Path("outputs/t2v-webrtc")
            recording_directory.mkdir(parents=True, exist_ok=True)
            self._artifact_path = recording_directory / f"{uuid4()}.mp4"
            self._artifact_output = Mp4VideoOutputTarget(
                output_path=self._artifact_path,
                fps=scenario.fps,
                output_layout=output_layout,
            )
            self._artifact_output.open()

    @property
    def step_index(self) -> int:
        """Return the index of the next autoregressive block."""
        return self._step_index

    @property
    def steady_output_frame_count(self) -> int:
        """Return the steady decoded frames produced by one iteration."""
        return self._steady_output_frame_count

    def next_step_request(self) -> StepRequest | None:
        """Stop shared serving after this session's requested video duration."""
        total_blocks = self._scenario.total_blocks
        if self._destroyed or (
            total_blocks is not None and self._step_index >= total_blocks
        ):
            return None
        return StepRequest(
            step_index=self._step_index,
            metadata={
                "steady_output_frame_count": self._steady_output_frame_count,
            },
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        """Generate and finalize one autoregressive video block."""
        if self._destroyed or self._cache is None:
            raise RuntimeError("Cannot generate from a destroyed T2VSession.")
        if inputs.global_conditioning:
            raise ValueError(
                "T2V global conditioning is fixed when the session is created."
            )
        if inputs.step:
            raise ValueError("This T2V application does not accept per-step input.")

        index = self._step_index
        video = self._pipeline.generate(
            autoregressive_index=index,
            cache=cast(Any, self._cache),
        )
        metrics = _metrics(
            self._pipeline.finalize(
                autoregressive_index=index,
                cache=cast(Any, self._cache),
            )
        )
        self._step_index += 1
        if not isinstance(video, torch.Tensor):
            raise TypeError(
                "T2V pipeline generate() must return torch.Tensor, got "
                f"{type(video).__name__}."
            )
        result = StepResult.from_video_chunk(
            step_index=index,
            video_chunk=video.detach(),
            layout=self._output_layout,
            metadata={FIELD_PROMPT: self._prompt},
            metrics=metrics,
        )
        if self._artifact_output is not None:
            self._artifact_output.write(result)
        return result

    def destroy(self) -> None:
        """Release this session's autoregressive cache."""
        if self._destroyed:
            return
        self._destroyed = True
        cache = self._cache
        self._cache = None
        artifact_output = self._artifact_output
        artifact_path = self._artifact_path
        self._artifact_output = None
        self._artifact_path = None
        try:
            if artifact_output is not None:
                artifacts = artifact_output.close()
                if (
                    artifacts
                    and artifact_path is not None
                    and self._record_artifact is not None
                ):
                    self._record_artifact(artifact_path, self._scenario)
        finally:
            close = getattr(cache, "close", None)
            if callable(close):
                close()


def _session_scenario(
    defaults: T2VSessionDefaults,
    initial_input: InferenceInput,
) -> T2VScenario:
    values = {
        FIELD_PROMPT: defaults.prompt,
        FIELD_TOTAL_BLOCKS: defaults.total_blocks,
        FIELD_PIXEL_HEIGHT: defaults.pixel_height,
        FIELD_PIXEL_WIDTH: defaults.pixel_width,
        FIELD_FPS: defaults.fps,
    }
    values.update(initial_input.global_conditioning)
    prompt = str(values[FIELD_PROMPT]).strip()
    if not prompt:
        raise ValueError("A non-empty text-to-video prompt is required.")
    values[FIELD_PROMPT] = prompt
    pixel_height = _positive_int(values[FIELD_PIXEL_HEIGHT], name=FIELD_PIXEL_HEIGHT)
    pixel_width = _positive_int(values[FIELD_PIXEL_WIDTH], name=FIELD_PIXEL_WIDTH)
    fps = _positive_int(values[FIELD_FPS], name=FIELD_FPS)
    total_blocks = values[FIELD_TOTAL_BLOCKS]
    if total_blocks is not None:
        if isinstance(total_blocks, bool) or not isinstance(total_blocks, int):
            raise TypeError(f"{FIELD_TOTAL_BLOCKS} must be an integer or None.")
        if total_blocks <= 0:
            raise ValueError(f"{FIELD_TOTAL_BLOCKS} must be > 0.")
    return T2VScenario(
        prompt=prompt,
        total_blocks=total_blocks,
        pixel_height=pixel_height,
        pixel_width=pixel_width,
        fps=fps,
    )


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


def _metrics(value: object) -> Mapping[str, float | int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "T2V pipeline finalize() must return a metrics mapping or None, got "
            f"{type(value).__name__}."
        )
    metrics: dict[str, float | int] = {}
    for key, metric in value.items():
        if not isinstance(key, str):
            raise TypeError("T2V pipeline metric keys must be strings.")
        if not isinstance(metric, (int, float)):
            raise TypeError(f"T2V pipeline metric {key!r} must be numeric.")
        metrics[key] = metric
    return metrics


__all__ = ["T2VScenario", "T2VSession", "T2VSessionDefaults"]
