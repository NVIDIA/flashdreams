# SPDX-FileCopyrightText: Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

"""Qwen3-VL presentations used to condition native MiniMax H3 inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from minimax_h3.constants import (
    TEXT_EMBED_DIM,
    TEXT_ENCODER_LAYER,
    TEXT_TAG,
    VIDEO_TAG,
)
from minimax_h3.reference_conditioning import (
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3Presentation:
    """Tokenized H3 presentation plus Qwen vision inputs."""

    token_ids: tuple[int, ...]
    token_tags: Tensor
    vision_inputs: dict[str, Tensor]

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("MiniMax H3 presentations must contain at least one token")
        if self.token_tags.ndim != 1 or self.token_tags.dtype != torch.long:
            raise ValueError("token_tags must be a one-dimensional torch.long tensor")
        if len(self.token_ids) != self.token_tags.numel():
            raise ValueError("token_tags must identify every presentation token")
        if any(
            type(token_id) is not int or token_id < 0
            for token_id in self.token_ids
        ):
            raise ValueError("token_ids must contain non-negative integers")
        if not bool(
            torch.isin(
                self.token_tags.detach().cpu(),
                torch.tensor([VIDEO_TAG, TEXT_TAG], dtype=torch.long),
            ).all()
        ):
            raise ValueError("presentation tags must be text or video")


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3TextCondition:
    """Encoded Qwen hidden state and the H3 tag of each row."""

    prompt_embeds: Tensor
    text_token_tags: Tensor

    def __post_init__(self) -> None:
        if (
            self.prompt_embeds.ndim != 3
            or self.prompt_embeds.shape[0] != 1
            or self.prompt_embeds.shape[2] != TEXT_EMBED_DIM
        ):
            raise ValueError(
                f"prompt_embeds must have shape [1, text_rows, {TEXT_EMBED_DIM}]"
            )
        if not self.prompt_embeds.is_floating_point() or not bool(
            torch.isfinite(self.prompt_embeds).all()
        ):
            raise ValueError("prompt_embeds must contain finite floating-point values")
        if (
            self.text_token_tags.ndim != 1
            or self.text_token_tags.dtype != torch.long
            or self.prompt_embeds.shape[1] != self.text_token_tags.numel()
        ):
            raise ValueError("text_token_tags must identify every prompt-embedding row")


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not isinstance(token_ids, list) or any(
        type(token_id) is not int for token_id in token_ids
    ):
        raise ValueError("tokenizer must return a list of integer input_ids")
    return token_ids


def _special_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if type(token_id) is not int or token_id < 0:
        raise ValueError(f"tokenizer does not define {token!r}")
    return token_id


def build_t2va_presentation(tokenizer: Any, prompt: str) -> MiniMaxH3Presentation:
    """Tokenize a text-only prompt verbatim, without chat or special tokens."""
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    token_ids = _tokenize(tokenizer, prompt)
    return MiniMaxH3Presentation(
        token_ids=tuple(token_ids),
        token_tags=torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long),
        vision_inputs={},
    )


def build_fl2va_presentation(
    tokenizer: Any,
    processor: Any,
    prompt: str,
    keyframes: Sequence[object],
) -> MiniMaxH3Presentation:
    """Build H3's labelled first/last-keyframe Qwen presentation.

    Args:
        tokenizer: Qwen2 tokenizer from the pinned H3 repository.
        processor: Qwen3-VL processor from the pinned H3 repository.
        prompt: Prompt appended verbatim after all keyframe blocks.
        keyframes: Zero, one, or two RGB keyframes in packed order.

    Returns:
        Token ids, H3 row tags, and batched image-processor tensors.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if len(keyframes) > 2:
        raise ValueError("FL2VA accepts at most a first and a last keyframe")
    if not keyframes:
        return build_t2va_presentation(tokenizer, prompt)

    image_features = processor.image_processor(
        images=list(keyframes), return_tensors="pt"
    )
    pixel_values = image_features["pixel_values"]
    image_grid_thw = image_features["image_grid_thw"]
    if not isinstance(pixel_values, Tensor) or not isinstance(image_grid_thw, Tensor):
        raise ValueError("Qwen image processor must return tensor vision features")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape != (len(keyframes), 3):
        raise ValueError("image_grid_thw must have shape [keyframes, 3]")
    merge_size = processor.image_processor.merge_size**2
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("Qwen image processor merge_size must be positive")

    vision_start = _special_token_id(tokenizer, "<|vision_start|>")
    image_pad = _special_token_id(tokenizer, "<|image_pad|>")
    vision_end = _special_token_id(tokenizer, "<|vision_end|>")
    token_ids: list[int] = []
    token_tags: list[int] = []
    for index, grid in enumerate(image_grid_thw):
        num_image_tokens = int(grid.prod()) // merge_size
        if num_image_tokens <= 0:
            raise ValueError("each keyframe must produce at least one vision token")
        label_ids = _tokenize(tokenizer, f"<Picture {index + 1}>: ")
        vision_ids = [vision_start] + [image_pad] * num_image_tokens + [vision_end]
        token_ids.extend(label_ids)
        token_ids.extend(vision_ids)
        token_tags.extend([TEXT_TAG] * len(label_ids))
        token_tags.extend([VIDEO_TAG] * len(vision_ids))
    prompt_ids = _tokenize(tokenizer, prompt)
    token_ids.extend(prompt_ids)
    token_tags.extend([TEXT_TAG] * len(prompt_ids))
    return MiniMaxH3Presentation(
        token_ids=tuple(token_ids),
        token_tags=torch.tensor(token_tags, dtype=torch.long),
        vision_inputs={
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        },
    )


def _sample_ref2va_video_frames(
    frames: np.ndarray,
    *,
    fps: float,
    sample_fps: float,
    temporal_patch: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Sample conditioner frames and timestamp merged vision blocks."""
    if frames.ndim != 4 or frames.shape[0] == 0 or frames.shape[3] != 3:
        raise ValueError("normalized video references must have shape [T, H, W, 3]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("reference video fps must be positive and finite")
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("video_sample_fps must be positive and finite")
    if type(temporal_patch) is not int or temporal_patch <= 0:
        raise ValueError("Qwen temporal_patch_size must be a positive integer")

    stride = fps / sample_fps
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        index = round(cursor)
        if not indices or index > indices[-1]:
            indices.append(index)
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"A reference video sampled at {sample_fps:g} fps needs at least "
            f"{minimum} frames at {fps:g} fps, got {frames.shape[0]}."
        )

    timestamps = [index / sample_fps for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    block_timestamps = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return [frames[index] for index in indices], block_timestamps


def build_ref2va_presentation(
    tokenizer: Any,
    processor: Any,
    prompt: str,
    references: Sequence[MiniMaxH3Reference],
    *,
    fps: float = 24.0,
    video_sample_fps: float = 2.0,
) -> MiniMaxH3Presentation:
    """Build H3's ordered multimodal REF2VA Qwen presentation.

    Image and video vision tensors are batched by modality, while labels stay
    in request order. A video soundtrack emits its audio label immediately
    before that video's label, matching the packed latent layout.

    Args:
        tokenizer: Qwen2 tokenizer from the pinned H3 repository.
        processor: Qwen3-VL processor from the pinned H3 repository.
        prompt: Prompt appended verbatim after all reference blocks.
        references: Normalized references in semantic packed order.
        fps: Rate carried by normalized reference videos.
        video_sample_fps: Rate at which Qwen observes reference video.

    Returns:
        Token ids, modality tags, and batched Qwen vision features.
    """
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not references:
        raise ValueError("REF2VA presentations require at least one reference")

    merge_size = processor.image_processor.merge_size**2
    if type(merge_size) is not int or merge_size <= 0:
        raise ValueError("Qwen image processor merge_size must be positive")
    vision_inputs: dict[str, Tensor] = {}

    images = [
        reference.image
        for reference in references
        if isinstance(reference, MiniMaxH3ImageReference)
    ]
    image_token_counts: list[int] = []
    if images:
        image_features = processor.image_processor(
            images=images, return_tensors="pt"
        )
        pixel_values = image_features["pixel_values"]
        image_grid_thw = image_features["image_grid_thw"]
        if not isinstance(pixel_values, Tensor) or not isinstance(
            image_grid_thw, Tensor
        ):
            raise ValueError("Qwen image processor must return tensor features")
        if image_grid_thw.shape != (len(images), 3):
            raise ValueError("image_grid_thw must identify every image reference")
        image_token_counts = [
            int(grid.prod()) // merge_size for grid in image_grid_thw
        ]
        if any(count <= 0 for count in image_token_counts):
            raise ValueError("each image reference must produce vision tokens")
        vision_inputs.update(
            pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )

    videos = [
        reference
        for reference in references
        if isinstance(reference, MiniMaxH3VideoReference)
    ]
    video_token_counts: list[int] = []
    video_timestamps: list[list[float]] = []
    if videos:
        temporal_patch = processor.video_processor.temporal_patch_size
        sampled = [
            _sample_ref2va_video_frames(
                reference.frames,
                fps=float(reference.fps if reference.fps is not None else fps),
                sample_fps=video_sample_fps,
                temporal_patch=temporal_patch,
            )
            for reference in videos
        ]
        video_timestamps = [timestamps for _, timestamps in sampled]
        video_features = processor.video_processor(
            videos=[np.stack(frames) for frames, _ in sampled],
            do_sample_frames=False,
            return_tensors="pt",
        )
        pixel_values_videos = video_features["pixel_values_videos"]
        video_grid_thw = video_features["video_grid_thw"]
        if not isinstance(pixel_values_videos, Tensor) or not isinstance(
            video_grid_thw, Tensor
        ):
            raise ValueError("Qwen video processor must return tensor features")
        if video_grid_thw.shape != (len(videos), 3):
            raise ValueError("video_grid_thw must identify every video reference")
        video_token_counts = [
            int(grid[1]) * int(grid[2]) // merge_size
            for grid in video_grid_thw
        ]
        if any(count <= 0 for count in video_token_counts):
            raise ValueError("each video vision block must produce tokens")
        for timestamps, grid in zip(video_timestamps, video_grid_thw):
            if int(grid[0]) != len(timestamps):
                raise ValueError(
                    "Qwen video blocks do not match MiniMax H3 timestamps"
                )
        vision_inputs.update(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    vision_start = _special_token_id(tokenizer, "<|vision_start|>")
    image_pad = _special_token_id(tokenizer, "<|image_pad|>")
    video_pad = _special_token_id(tokenizer, "<|video_pad|>")
    vision_end = _special_token_id(tokenizer, "<|vision_end|>")
    token_ids: list[int] = []
    token_tags: list[int] = []

    def emit_text(value: str) -> None:
        ids = _tokenize(tokenizer, value)
        token_ids.extend(ids)
        token_tags.extend([TEXT_TAG] * len(ids))

    def emit_vision(pad_token: int, num_tokens: int) -> None:
        ids = [vision_start] + [pad_token] * num_tokens + [vision_end]
        token_ids.extend(ids)
        token_tags.extend([VIDEO_TAG] * len(ids))

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit_text(f"<Audio {counts['audio']}>: ")
        if isinstance(reference, MiniMaxH3ImageReference):
            counts["image"] += 1
            emit_text(f"<Picture {counts['image']}>: ")
            emit_vision(image_pad, image_token_counts[counts["image"] - 1])
        elif isinstance(reference, MiniMaxH3VideoReference):
            counts["video"] += 1
            emit_text(f"<Video {counts['video']}>: ")
            for timestamp in video_timestamps[counts["video"] - 1]:
                emit_text(f"<{timestamp:.1f} seconds>")
                emit_vision(video_pad, video_token_counts[counts["video"] - 1])
    emit_text(prompt)
    return MiniMaxH3Presentation(
        token_ids=tuple(token_ids),
        token_tags=torch.tensor(token_tags, dtype=torch.long),
        vision_inputs=vision_inputs,
    )


@torch.no_grad()
def encode_presentation(
    text_encoder: Any,
    processor: Any,
    presentation: MiniMaxH3Presentation,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    text_encoder_layer: int = TEXT_ENCODER_LAYER,
) -> MiniMaxH3TextCondition:
    """Encode one H3 presentation with Qwen3-VL decoder layer 50.

    Calling the conditioner submodel avoids the unused vocabulary-wide
    language-model projection. The owner must place that submodel on
    ``device`` before this call; this function does not own offload policy.
    """
    if type(text_encoder_layer) is not int or text_encoder_layer < 0:
        raise ValueError("text_encoder_layer must be a non-negative integer")
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers <= text_encoder_layer:
        raise ValueError(
            f"H3 needs hidden_states[{text_encoder_layer}], but the Qwen "
            f"conditioner has only {num_layers} layers"
        )
    target_device = torch.device(device)
    encoder_dtype = text_encoder.dtype
    input_ids = torch.tensor(
        [presentation.token_ids], dtype=torch.long, device=target_device
    )
    mm_token_type_ids = torch.tensor(
        processor.create_mm_token_type_ids([list(presentation.token_ids)]),
        dtype=torch.long,
        device=target_device,
    )
    vision_inputs = {}
    for name, value in presentation.vision_inputs.items():
        target_dtype = encoder_dtype if name.startswith("pixel_") else value.dtype
        vision_inputs[name] = value.to(device=target_device, dtype=target_dtype)

    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        **vision_inputs,
    )
    if len(outputs.hidden_states) <= text_encoder_layer:
        raise RuntimeError("Qwen conditioner returned too few hidden states")
    prompt_embeds = outputs.hidden_states[text_encoder_layer].to(
        device=target_device, dtype=dtype or encoder_dtype
    )
    return MiniMaxH3TextCondition(
        prompt_embeds=prompt_embeds,
        text_token_tags=presentation.token_tags.to(target_device),
    )


__all__ = [
    "MiniMaxH3Presentation",
    "MiniMaxH3TextCondition",
    "build_fl2va_presentation",
    "build_ref2va_presentation",
    "build_t2va_presentation",
    "encode_presentation",
]
