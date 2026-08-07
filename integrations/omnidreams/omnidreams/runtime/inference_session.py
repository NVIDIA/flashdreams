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

"""OmniDreams inference session with embedding and HDMap conditions."""

from typing import Annotated, TypeAlias, cast

from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.runtime.builtin.inference_output.frame_chunk import FrameChunkOutput
from flashdreams.runtime.inference_session import (
    InferenceGlobalCondition as BaseInferenceGlobalCondition,
)
from flashdreams.runtime.inference_session import InferenceInput as BaseInferenceInput
from flashdreams.runtime.inference_session import (
    InferenceSession as BaseInferenceSession,
)
from flashdreams.runtime.inference_session import (
    InferenceUserCondition as BaseInferenceUserCondition,
)
from omnidreams.pipeline import OmnidreamsPipeline, OmnidreamsPipelineCache
from pydantic import AfterValidator, Field, TypeAdapter, ValidationInfo
from torch import Tensor
from typing_extensions import TypedDict


def _validate_tensor_shape(
    tensor: Tensor,
    expected_shape: tuple[int | None, ...],
    shape_description: str,
) -> Tensor:
    """Validate a tensor's rank, fixed axes, and non-empty dimensions."""
    expected_rank = len(expected_shape)
    if tensor.ndim != expected_rank:
        raise ValueError(
            f"expected a rank-{expected_rank} tensor; got rank-{tensor.ndim} "
            f"with shape {tuple(tensor.shape)}"
        )

    if any(
        expected_size is not None and tensor.shape[axis] != expected_size
        for axis, expected_size in enumerate(expected_shape)
    ):
        raise ValueError(
            f"expected tensor shape {shape_description}; got {tuple(tensor.shape)}"
        )

    if any(size <= 0 for size in tensor.shape):
        raise ValueError(
            f"expected every axis in tensor shape {shape_description} to be positive; "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


def _validate_hdmap(tensor: Tensor) -> Tensor:
    return _validate_tensor_shape(
        tensor,
        (None, None, None, 3, None, None),
        "[B, V, T, 3, H, W]",
    )


def _validate_text_embeddings(tensor: Tensor) -> Tensor:
    return _validate_tensor_shape(
        tensor,
        (None, None, None, None),
        "[B, V, L, D]",
    )


def _validate_image_embeddings(tensor: Tensor) -> Tensor:
    return _validate_tensor_shape(
        tensor,
        (None, None, 1, None, None, None),
        "[B, V, 1, Cl, Hl, Wl]",
    )


_HDMapTensor: TypeAlias = Annotated[Tensor, AfterValidator(_validate_hdmap)]
_TextEmbeddingsTensor: TypeAlias = Annotated[
    Tensor, AfterValidator(_validate_text_embeddings)
]
_ImageEmbeddingsTensor: TypeAlias = Annotated[
    Tensor, AfterValidator(_validate_image_embeddings)
]


class InferenceUserCondition(BaseInferenceUserCondition):
    """Per-step HDMap condition for OmniDreams inference."""

    hdmap: _HDMapTensor
    """HDMap pixels ``[B, V, T, 3, H, W]`` for the next video chunk."""


class InferenceGlobalCondition(BaseInferenceGlobalCondition):
    """Rollout-wide embedding conditions for OmniDreams inference."""

    text_embeddings: _TextEmbeddingsTensor
    """Text embeddings ``[B, V, L, D]`` for the rollout prompts."""

    negative_text_embeddings: _TextEmbeddingsTensor | None = None
    """Optional negative-prompt embeddings ``[B, V, L, D]`` used for CFG."""

    image_embeddings: _ImageEmbeddingsTensor
    """First-frame image embeddings ``[B, V, 1, Cl, Hl, Wl]``."""


InferenceInput: TypeAlias = BaseInferenceInput[
    InferenceUserCondition, InferenceGlobalCondition
]
"""OmniDreams conditions consumed by one inference step."""


class _InferenceValidationContext(TypedDict):
    """Pipeline-dependent state supplied to Pydantic input validation."""

    pipeline: OmnidreamsPipeline
    """Pipeline whose shape contracts apply to the input."""

    autoregressive_index: int
    """Index of the step being validated."""

    rollout_resolution: tuple[int, int] | None
    """Pixel resolution established by the active rollout, if any."""


def _validate_condition_shapes(
    inference_input: InferenceInput,
    validation_info: ValidationInfo,
) -> InferenceInput:
    """Validate shape relationships between per-step and rollout conditions."""
    global_condition = inference_input.global_condition
    hdmap = inference_input.user_condition.hdmap

    if global_condition is not None:
        text_embeddings = global_condition.text_embeddings
        image_embeddings = global_condition.image_embeddings
        batch_view_shapes = {
            "hdmap": tuple(hdmap.shape[:2]),
            "text_embeddings": tuple(text_embeddings.shape[:2]),
            "image_embeddings": tuple(image_embeddings.shape[:2]),
        }
        if len(set(batch_view_shapes.values())) != 1:
            raise ValueError(
                "expected hdmap, text_embeddings, and image_embeddings to share "
                f"[B, V] dimensions; got {batch_view_shapes}"
            )

        negative_text_embeddings = global_condition.negative_text_embeddings
        if (
            negative_text_embeddings is not None
            and negative_text_embeddings.shape != text_embeddings.shape
        ):
            raise ValueError(
                "expected negative_text_embeddings shape to match text_embeddings; "
                f"got {tuple(negative_text_embeddings.shape)} and "
                f"{tuple(text_embeddings.shape)}"
            )

    if validation_info.context is None:
        return inference_input

    context = cast(_InferenceValidationContext, validation_info.context)
    pipeline = context["pipeline"]
    autoregressive_index = context["autoregressive_index"]
    rollout_resolution = context["rollout_resolution"]

    pipeline._validate_image_resolution(hdmap)

    actual_frames = int(hdmap.shape[2])
    expected_frames = pipeline.get_num_frames(autoregressive_index)
    if actual_frames != expected_frames:
        raise ValueError(
            f"expected hdmap T={expected_frames} at autoregressive index "
            f"{autoregressive_index}; got T={actual_frames}"
        )

    hdmap_resolution = (int(hdmap.shape[-2]), int(hdmap.shape[-1]))
    if rollout_resolution is not None and hdmap_resolution != rollout_resolution:
        raise ValueError(
            f"expected hdmap resolution {rollout_resolution} for the active rollout; "
            f"got {hdmap_resolution}"
        )

    if global_condition is not None:
        decoder = pipeline.decoder
        assert isinstance(decoder, StreamingVideoDecoder)
        compression = decoder.spatial_compression_ratio
        expected_latent_resolution = (
            hdmap_resolution[0] // compression,
            hdmap_resolution[1] // compression,
        )
        image_embeddings = global_condition.image_embeddings
        image_latent_resolution = (
            int(image_embeddings.shape[-2]),
            int(image_embeddings.shape[-1]),
        )
        if image_latent_resolution != expected_latent_resolution:
            raise ValueError(
                "expected image_embeddings latent resolution "
                f"{expected_latent_resolution} for hdmap resolution "
                f"{hdmap_resolution}; got {image_latent_resolution}"
            )
    return inference_input


_ValidatedInferenceInput: TypeAlias = Annotated[
    InferenceInput, AfterValidator(_validate_condition_shapes)
]

_INFERENCE_INPUT_ADAPTER = TypeAdapter(_ValidatedInferenceInput)

_PresentationFps: TypeAlias = Annotated[float, Field(gt=0, allow_inf_nan=False)]
_PRESENTATION_FPS_ADAPTER = TypeAdapter(_PresentationFps)


class InferenceSession(BaseInferenceSession):
    """Stateful OmniDreams inference session backed by a per-rollout cache."""

    pipeline: OmnidreamsPipeline
    """OmniDreams pipeline shared with the inference runtime."""

    cache: OmnidreamsPipelineCache | None
    """Per-rollout cache; ``None`` until global conditions initialize it."""

    autoregressive_index: int
    """Zero-based index assigned to the next inference step."""

    presentation_fps: _PresentationFps
    """Frame rate used for output presentation timestamps."""

    _rollout_resolution: tuple[int, int] | None
    """HDMap pixel resolution fixed by the first successful rollout step."""

    _presented_frame_count: int
    """Number of frames emitted on the current presentation timeline."""

    def __init__(
        self,
        pipeline: OmnidreamsPipeline,
        *,
        presentation_fps: _PresentationFps = 30.0,
    ) -> None:
        """Initialize the session with a presentation frame rate.

        Args:
            pipeline: OmniDreams pipeline to drive.
            presentation_fps: Frame rate for output presentation timestamps.

        Raises:
            ValidationError: ``presentation_fps`` is not positive and finite.
        """
        self.presentation_fps = _PRESENTATION_FPS_ADAPTER.validate_python(
            presentation_fps
        )
        super().__init__(pipeline)

    def reset(self) -> None:
        """Reset the session to await rollout-wide embedding conditions."""
        self.cache = None
        self.autoregressive_index = 0
        self._rollout_resolution = None
        self._presented_frame_count = 0

    def step(self, inference_input: InferenceInput) -> FrameChunkOutput:
        """Generate one video chunk from validated OmniDreams conditions.

        Args:
            inference_input: Per-step HDMap and optional first-step embeddings.

        Returns:
            Decoded video chunk for the current autoregressive step.

        Raises:
            ValueError: Global conditions are missing on the first step or are
                supplied after the rollout cache has been initialized.
            ValidationError: ``inference_input`` fails Pydantic validation.
        """
        inference_input = _INFERENCE_INPUT_ADAPTER.validate_python(
            inference_input,
            context=_InferenceValidationContext(
                pipeline=self.pipeline,
                autoregressive_index=self.autoregressive_index,
                rollout_resolution=self._rollout_resolution,
            ),
        )
        global_condition = inference_input.global_condition
        if self.cache is None:
            if global_condition is None:
                raise ValueError(
                    "global_condition is required on the first step after reset()."
                )
            self.cache = self.pipeline.initialize_cache_from_embeddings(
                text_embeddings=global_condition.text_embeddings,
                image_embeddings=global_condition.image_embeddings,
                negative_text_embeddings=global_condition.negative_text_embeddings,
            )
            hdmap = inference_input.user_condition.hdmap
            self._rollout_resolution = (
                int(hdmap.shape[-2]),
                int(hdmap.shape[-1]),
            )
        elif global_condition is not None:
            # TODO: Support in rollout global condition modification.
            raise ValueError(
                "global_condition can only be supplied on the first step after reset()."
            )

        video = self.pipeline.generate(
            autoregressive_index=self.autoregressive_index,
            cache=self.cache,
            hdmap=inference_input.user_condition.hdmap,
        )
        self.pipeline.finalize(
            autoregressive_index=self.autoregressive_index,
            cache=self.cache,
        )
        start_timestamp = self._presented_frame_count / self.presentation_fps
        self._presented_frame_count += int(video.shape[2])
        self.autoregressive_index += 1
        return FrameChunkOutput(
            value=video,
            start_timestamp=start_timestamp,
            fps=self.presentation_fps,
        )
