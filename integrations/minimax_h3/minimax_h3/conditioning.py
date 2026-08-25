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

"""Native MiniMax H3 packed-sequence conditioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from minimax_h3.constants import (
    AUDIO_CHANNELS,
    AUDIO_LATENT_CHANNELS,
    AUDIO_LATENTS_PER_SECOND,
    AUDIO_TAG,
    CANVAS_MAX_PIXELS,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    FPS,
    FRAME_CHUNK,
    FRAME_REMAINDER,
    KEYFRAME_NOISE_AUG,
    MAX_DURATION,
    MIN_DURATION,
    PATCH_SIZE,
    VAE_LATENT_CHANNELS,
    VAE_SPATIAL_COMPRESSION_RATIO,
    VIDEO_TAG,
    align_num_frames,
    validate_canvas,
)
from minimax_h3.model import MiniMaxH3DenoiseState
from minimax_h3.reference_conditioning import (
    MiniMaxH3AudioReference,
    MiniMaxH3EncodedReferences,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3PackedLayout:
    """One H3 transformer sequence and its modality row partitions."""

    position_ids: Tensor
    token_tags: Tensor
    video_indices: Tensor
    audio_indices: Tensor
    text_indices: Tensor
    num_condition_video_rows: int
    num_condition_audio_rows: int


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    *,
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
    multiple: int = CANVAS_MULTIPLE,
) -> tuple[int, int]:
    """Resolve an aspect ratio to the released H3 canvas policy.

    Args:
        aspect_width: Positive width of the requested aspect ratio.
        aspect_height: Positive height of the requested aspect ratio.
        short_edge: Target size of the shorter edge before the area cap.
        max_pixels: Maximum pre-rounding canvas area.
        multiple: Multiple to which both axes are rounded.

    Returns:
        Resolved ``(height, width)``.

    Raises:
        ValueError: The ratio, area policy, or multiple is invalid.
    """
    if not np.isfinite(aspect_width) or not np.isfinite(aspect_height):
        raise ValueError("aspect dimensions must be finite")
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError("aspect dimensions must be positive")
    if type(short_edge) is not int or short_edge <= 0:
        raise ValueError("short_edge must be a positive integer")
    if type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("max_pixels must be a positive integer")
    if type(multiple) is not int or multiple <= 0:
        raise ValueError("multiple must be a positive integer")

    ratio = aspect_width / aspect_height
    if not 0.25 <= ratio <= 4.0:
        raise ValueError("aspect ratio must be between 1:4 and 4:1")
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def video_latent_num_frames(num_frames: int) -> int:
    """Return the H3 video-latent length for an aligned frame count."""
    if type(num_frames) is not int or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    if num_frames % FRAME_CHUNK != FRAME_REMAINDER:
        raise ValueError("num_frames must be aligned to the H3 frame grid")
    return (num_frames - FRAME_REMAINDER) // FRAME_CHUNK * FRAME_REMAINDER + 2


def audio_latent_num_frames(num_frames: int) -> int:
    """Return the 40 Hz audio-latent length covering ``num_frames``."""
    if type(num_frames) is not int or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    return round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND)


def patchify_video_latents(
    latents: Tensor, patch_size: tuple[int, int, int] = PATCH_SIZE
) -> Tensor:
    """Pack ``[batch, channels, time, height, width]`` latents into rows."""
    if latents.ndim != 5:
        raise ValueError("video latents must have rank 5")
    if len(patch_size) != 3 or any(
        type(size) is not int or size <= 0 for size in patch_size
    ):
        raise ValueError("patch_size must contain 3 positive integers")
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by "
            f"patch {patch_size}."
        )
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(
        -1, channels * patch_t * patch_h * patch_w
    ).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False)
    return torch.from_numpy(grid * _ROPE_SPATIAL_SCALE).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> Tensor:
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE
            * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[Tensor, Tensor]:
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    frame_grid = torch.stack([grid.reshape(-1) for grid in grids], dim=-1)
    return frame_grid, width_grid


def build_packed_layout(
    text_token_tags: Tensor,
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    keyframe_anchors: Sequence[str] = (),
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> MiniMaxH3PackedLayout:
    """Build H3's ``text | conditions | audio | video`` row layout.

    Args:
        text_token_tags: One H3 modality tag for each prompt-embedding row.
        num_latent_frames: Number of generated video latent frames.
        latent_height: Generated video latent height.
        latent_width: Generated video latent width.
        num_audio_latents: Generated audio latent steps per channel.
        keyframe_anchors: ``"first"`` or ``"last"`` for each keyframe.
        patch_size: Released transformer video patch geometry.

    Returns:
        Packed layout with float64 rotary coordinates and row partitions.
    """
    if text_token_tags.ndim != 1:
        raise ValueError("text_token_tags must be one-dimensional")
    dimensions = {
        "num_latent_frames": num_latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "num_audio_latents": num_audio_latents,
    }
    for name, value in dimensions.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if patch_size != PATCH_SIZE:
        raise ValueError(f"released MiniMax H3 requires patch_size={PATCH_SIZE}")
    _, patch_h, patch_w = patch_size
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError("latent canvas must be divisible by the spatial patch")

    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = (
        num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
    )
    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(
        num_text_tokens, dtype=torch.float64
    )
    frame_grid, width_grid = _frame_position_grid(
        latent_height, latent_width, patch_h, patch_w
    )
    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            spans = np.ones(num_latent_frames, dtype=np.float64)
            spans *= _ROPE_FRAME_RESCALE
            for offset, frames_per_latent in enumerate(_ROPE_FRAMES_PER_LATENT):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= frames_per_latent
            anchor_time = (
                float(num_text_tokens)
                + float(spans.sum())
                - _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(
                f"keyframe anchors must be 'first' or 'last', got {anchor!r}"
            )
        rows = slice(
            condition_start + index * rows_per_frame,
            condition_start + (index + 1) * rows_per_frame,
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_tokens) + torch.arange(
        num_audio_latents, dtype=torch.float64
    )
    position_ids[audio_start:video_start, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full(
                (num_audio_latents,), float(width_grid[0]), dtype=torch.float64
            ),
            torch.full(
                (num_audio_latents,), float(width_grid[-1]), dtype=torch.float64
            ),
        ]
    )

    video_positions = torch.empty(
        num_latent_frames, rows_per_frame, 3, dtype=torch.float64
    )
    video_positions[:, :, 0] = _temporal_position_grid(
        num_latent_frames, float(num_text_tokens)
    )[:, None]
    video_positions[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_positions.reshape(-1, 3)

    video_indices = torch.cat(
        [
            torch.arange(condition_start, audio_start),
            torch.arange(video_start, sequence_length),
        ]
    )
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return MiniMaxH3PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def _fill_audio_positions(
    position_ids: Tensor,
    rows: slice,
    num_audio_latents: int,
    origin: float,
    width_grid: Tensor,
) -> None:
    """Place channel-major stereo rows on time and opposite canvas edges."""
    if num_audio_latents == 0:
        return
    audio_time = origin + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full(
                (num_audio_latents,),
                float(width_grid[0]),
                dtype=torch.float64,
            ),
            torch.full(
                (num_audio_latents,),
                float(width_grid[-1]),
                dtype=torch.float64,
            ),
        ]
    )


def build_ref2va_packed_layout(
    text_token_tags: Tensor,
    references: Sequence[MiniMaxH3Reference],
    encoded: MiniMaxH3EncodedReferences,
    *,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> MiniMaxH3PackedLayout:
    """Build REF2VA's text, ordered references, audio, and video layout.

    Every reference advances one shared float64 rotary clock. A video's clean
    soundtrack rows immediately precede its visual rows and share their time
    origin; the next reference starts after the longer of those two streams.

    Args:
        text_token_tags: Modality tag for each Qwen presentation row.
        references: Normalized references in semantic request order.
        encoded: Visual and audio conditioning filtered from that order.
        num_latent_frames: Number of generated video latent frames.
        latent_height: Generated video latent height.
        latent_width: Generated video latent width.
        num_audio_latents: Generated audio latent steps per channel.
        patch_size: Released H3 transformer video patch.

    Returns:
        Packed float64 coordinates, modality partitions, and condition counts.

    Raises:
        ValueError: Reference kinds, encoded rows, or target geometry mismatch.
    """
    if text_token_tags.ndim != 1:
        raise ValueError("text_token_tags must be one-dimensional")
    dimensions = {
        "num_latent_frames": num_latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "num_audio_latents": num_audio_latents,
    }
    if any(type(value) is not int or value <= 0 for value in dimensions.values()):
        raise ValueError("REF2VA target latent dimensions must be positive integers")
    if patch_size != PATCH_SIZE:
        raise ValueError(f"released MiniMax H3 requires patch_size={PATCH_SIZE}")
    _, patch_h, patch_w = patch_size
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError("target latent canvas must be divisible by the spatial patch")
    if not references:
        raise ValueError("REF2VA layout requires at least one reference")
    allowed = (
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
        MiniMaxH3AudioReference,
    )
    if any(not isinstance(reference, allowed) for reference in references):
        raise ValueError("REF2VA layout received an unsupported reference type")
    num_visual = sum(
        isinstance(reference, (MiniMaxH3ImageReference, MiniMaxH3VideoReference))
        for reference in references
    )
    num_audio = sum(reference.has_audio for reference in references)
    if len(encoded.video) != num_visual:
        raise ValueError("encoded video references do not match the request")
    if len(encoded.audio) != num_audio:
        raise ValueError("encoded audio references do not match the request")

    visual_geometry: list[tuple[int, int, int]] = []
    visual_index = 0
    for reference in references:
        if not isinstance(
            reference, (MiniMaxH3ImageReference, MiniMaxH3VideoReference)
        ):
            continue
        latents = encoded.video[visual_index]
        visual_index += 1
        if (
            latents.ndim != 5
            or latents.shape[0] != 1
            or latents.shape[1] != VAE_LATENT_CHANNELS
            or any(size <= 0 for size in latents.shape[2:])
        ):
            raise ValueError("encoded reference video latents are malformed")
        if not latents.is_floating_point() or not bool(
            torch.isfinite(latents).all()
        ):
            raise ValueError(
                "encoded reference video latents must be finite floating point"
            )
        frames, height, width = tuple(latents.shape[2:])
        if height % patch_h or width % patch_w:
            raise ValueError("reference latent canvas must align to the spatial patch")
        if isinstance(reference, MiniMaxH3ImageReference) and frames != 1:
            raise ValueError("an image reference must encode to one latent frame")
        visual_geometry.append((frames, height, width))

    audio_row_counts: list[int] = []
    for rows in encoded.audio:
        if (
            rows.ndim != 2
            or rows.shape[0] == 0
            or rows.shape[0] % AUDIO_CHANNELS
            or rows.shape[1] != AUDIO_LATENT_CHANNELS
        ):
            raise ValueError("encoded reference audio rows are malformed")
        if not rows.is_floating_point() or not bool(torch.isfinite(rows).all()):
            raise ValueError(
                "encoded reference audio rows must be finite floating point"
            )
        audio_row_counts.append(rows.shape[0])

    num_text_tokens = text_token_tags.shape[0]
    target_video_rows = (
        num_latent_frames
        * (latent_height // patch_h)
        * (latent_width // patch_w)
    )
    target_audio_rows = num_audio_latents * AUDIO_CHANNELS
    condition_video_rows = sum(
        frames * (height // patch_h) * (width // patch_w)
        for frames, height, width in visual_geometry
    )
    condition_audio_rows = sum(audio_row_counts)
    sequence_length = (
        num_text_tokens
        + condition_video_rows
        + condition_audio_rows
        + target_audio_rows
        + target_video_rows
    )
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(
        num_text_tokens, dtype=torch.float64
    )
    target_frame_grid, target_width_grid = _frame_position_grid(
        latent_height, latent_width, patch_h, patch_w
    )

    video_index_parts: list[Tensor] = []
    audio_index_parts: list[Tensor] = []
    visual_iter = iter(visual_geometry)
    audio_iter = iter(audio_row_counts)
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference in references:
        if isinstance(reference, MiniMaxH3ImageReference):
            frames, height, width = next(visual_iter)
            num_rows = frames * (height // patch_h) * (width // patch_w)
            rows = slice(cursor, cursor + num_rows)
            cursor = rows.stop
            video_index_parts.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(
                height, width, patch_h, patch_w
            )
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            rotary_time += 1.0
        elif isinstance(reference, MiniMaxH3AudioReference):
            num_rows = next(audio_iter)
            reference_steps = num_rows // AUDIO_CHANNELS
            rows = slice(cursor, cursor + num_rows)
            cursor = rows.stop
            audio_index_parts.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(
                position_ids,
                rows,
                reference_steps,
                rotary_time,
                target_width_grid,
            )
            rotary_time += float(reference_steps)
        elif isinstance(reference, MiniMaxH3VideoReference):
            num_audio_rows = next(audio_iter) if reference.has_audio else 0
            reference_steps = num_audio_rows // AUDIO_CHANNELS
            frames, height, width = next(visual_iter)
            num_video_rows = frames * (height // patch_h) * (width // patch_w)
            audio_rows = slice(cursor, cursor + num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
            cursor = video_rows.stop
            audio_index_parts.append(
                torch.arange(audio_rows.start, audio_rows.stop)
            )
            video_index_parts.append(
                torch.arange(video_rows.start, video_rows.stop)
            )

            frame_grid, width_grid = _frame_position_grid(
                height, width, patch_h, patch_w
            )
            _fill_audio_positions(
                position_ids,
                audio_rows,
                reference_steps,
                rotary_time,
                width_grid,
            )
            frame_time = _temporal_position_grid(frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(
                frame_grid.shape[0]
            )
            position_ids[video_rows, 1:] = frame_grid.repeat(frames, 1)
            video_span = sum(
                _ROPE_FRAME_RESCALE
                * _ROPE_FRAMES_PER_LATENT[
                    index % len(_ROPE_FRAMES_PER_LATENT)
                ]
                for index in range(frames)
            )
            rotary_time += max(float(reference_steps), video_span)

    audio_start = cursor
    video_start = audio_start + target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(
        target_frame_grid.shape[0]
    )
    position_ids[video_start:, 1:] = target_frame_grid.repeat(
        num_latent_frames, 1
    )

    video_indices = torch.cat(
        video_index_parts + [torch.arange(video_start, sequence_length)]
    )
    audio_indices = torch.cat(
        audio_index_parts + [torch.arange(audio_start, video_start)]
    )
    text_indices = torch.arange(num_text_tokens)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return MiniMaxH3PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=condition_video_rows,
        num_condition_audio_rows=condition_audio_rows,
    )


def _randn_tensor(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator | None,
    device: torch.device,
) -> Tensor:
    """Draw like the pinned oracle, including CPU-generator handoff."""
    random_device = device
    if generator is not None and generator.device.type != device.type:
        if generator.device.type != "cpu":
            raise ValueError(
                f"cannot draw a {device.type} tensor from a "
                f"{generator.device.type} generator"
            )
        random_device = torch.device("cpu")
    return torch.randn(
        shape,
        generator=generator,
        device=random_device,
        dtype=torch.float32,
    ).to(device)


def prepare_denoise_state(
    prompt_embeds: Tensor,
    text_token_tags: Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    generator: torch.Generator | None = None,
    video_noise: Tensor | None = None,
    audio_noise: Tensor | None = None,
    condition_latents: Sequence[Tensor] = (),
    keyframe_anchors: Sequence[str] = (),
    device: torch.device | str | None = None,
) -> MiniMaxH3DenoiseState:
    """Prepare a native T2VA or FL2VA request for joint denoising.

    Conditioning noise is drawn first, one keyframe at a time, followed by
    generated video noise and channel-major audio noise. Supplying either
    generated noise tensor skips only that draw, matching the released RNG
    stream contract.

    Args:
        prompt_embeds: Qwen3-VL hidden state shaped ``[1, text_rows, 5120]``.
        text_token_tags: H3 modality tag for each prompt-embedding row.
        num_frames: Already aligned ``17k + 5`` output frame count.
        height: Output canvas height in pixels.
        width: Output canvas width in pixels.
        generator: Request-owned random generator.
        video_noise: Optional generated video noise in unpatchified form.
        audio_noise: Optional stereo audio noise shaped ``[2, 32, time]``.
        condition_latents: Normalized keyframe latents in packed order.
        keyframe_anchors: ``"first"`` or ``"last"`` for each keyframe.
        device: Device on which to prepare latent rows.

    Returns:
        Complete typed state for :meth:`MiniMaxH3DiffusionModel.generate_joint`.

    Raises:
        ValueError: The presentation, geometry, or supplied latents mismatch.
    """
    validate_canvas(width, height)
    min_num_frames = align_num_frames(MIN_DURATION)
    max_num_frames = align_num_frames(MAX_DURATION)
    if not min_num_frames <= num_frames <= max_num_frames:
        raise ValueError(
            f"num_frames must be an aligned H3 frame count from "
            f"{min_num_frames} through {max_num_frames}"
        )
    num_latent_frames = video_latent_num_frames(num_frames)
    latent_height = height // VAE_SPATIAL_COMPRESSION_RATIO
    latent_width = width // VAE_SPATIAL_COMPRESSION_RATIO
    num_audio_latents = audio_latent_num_frames(num_frames)
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
        raise ValueError("prompt_embeds must have shape [1, text_rows, hidden]")
    if (
        text_token_tags.ndim != 1
        or prompt_embeds.shape[1] != text_token_tags.numel()
    ):
        raise ValueError("text_token_tags must identify every prompt-embedding row")
    if len(condition_latents) != len(keyframe_anchors):
        raise ValueError("each condition latent requires one keyframe anchor")

    target_device = torch.device(device or prompt_embeds.device)
    expected_condition_shape = (
        1,
        VAE_LATENT_CHANNELS,
        1,
        latent_height,
        latent_width,
    )
    condition_rows = []
    noise_level = torch.tensor(
        KEYFRAME_NOISE_AUG, dtype=torch.float32, device=target_device
    )
    for condition in condition_latents:
        if tuple(condition.shape) != expected_condition_shape:
            raise ValueError(
                "keyframe condition latents must match the target latent canvas"
            )
        clean = condition.to(target_device, torch.float32)
        noise = _randn_tensor(
            tuple(clean.shape), generator=generator, device=target_device
        )
        noised = noise_level * clean + (1.0 - noise_level) * noise
        condition_rows.append(patchify_video_latents(noised))

    expected_video_shape = (
        1,
        VAE_LATENT_CHANNELS,
        num_latent_frames,
        latent_height,
        latent_width,
    )
    if video_noise is None:
        video_noise = _randn_tensor(
            expected_video_shape, generator=generator, device=target_device
        )
    elif tuple(video_noise.shape) != expected_video_shape:
        raise ValueError(f"video_noise must have shape {expected_video_shape}")
    video_rows = patchify_video_latents(
        video_noise.to(target_device, torch.float32)
    )
    if condition_rows:
        video_rows = torch.cat(condition_rows + [video_rows])

    expected_audio_shape = (
        AUDIO_CHANNELS,
        AUDIO_LATENT_CHANNELS,
        num_audio_latents,
    )
    if audio_noise is None:
        audio_rows = _randn_tensor(
            (num_audio_latents * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
            generator=generator,
            device=target_device,
        )
    else:
        if tuple(audio_noise.shape) != expected_audio_shape:
            raise ValueError(f"audio_noise must have shape {expected_audio_shape}")
        audio_rows = (
            audio_noise.to(target_device, torch.float32)
            .permute(0, 2, 1)
            .reshape(-1, AUDIO_LATENT_CHANNELS)
        )

    layout = build_packed_layout(
        text_token_tags,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        keyframe_anchors=keyframe_anchors,
    )
    return MiniMaxH3DenoiseState(
        latents=video_rows,
        audio_latents=audio_rows,
        prompt_embeds=prompt_embeds.to(target_device),
        position_ids=layout.position_ids.to(target_device),
        token_tags=layout.token_tags.to(target_device),
        video_indices=layout.video_indices.to(target_device),
        audio_indices=layout.audio_indices.to(target_device),
        text_indices=layout.text_indices.to(target_device),
        num_condition_video_rows=layout.num_condition_video_rows,
        num_condition_audio_rows=layout.num_condition_audio_rows,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        audio_channels=AUDIO_CHANNELS,
    )


def prepare_ref2va_denoise_state(
    prompt_embeds: Tensor,
    text_token_tags: Tensor,
    references: Sequence[MiniMaxH3Reference],
    encoded: MiniMaxH3EncodedReferences,
    *,
    num_frames: int,
    height: int,
    width: int,
    generator: torch.Generator | None = None,
    video_noise: Tensor | None = None,
    audio_noise: Tensor | None = None,
    device: torch.device | str | None = None,
) -> MiniMaxH3DenoiseState:
    """Prepare ordered REF2VA references and generated rows for denoising.

    Visual reference noise is drawn first, one reference at a time, followed
    by generated video and audio. Clean soundtrack rows consume no randomness
    and are prepended to the generated audio rows.

    Args:
        prompt_embeds: Qwen3-VL hidden state shaped ``[1, text_rows, 5120]``.
        text_token_tags: H3 modality tag for every presentation row.
        references: Normalized references in semantic request order.
        encoded: Native VAE conditioning for the same references.
        num_frames: Final aligned output frame count.
        height: Output canvas height in pixels.
        width: Output canvas width in pixels.
        generator: Request-owned random generator.
        video_noise: Optional generated video noise in unpatchified form.
        audio_noise: Optional stereo audio noise shaped ``[2, 32, time]``.
        device: Device on which to prepare packed rows.

    Returns:
        Complete typed state for joint native H3 denoising.

    Raises:
        ValueError: Presentation, output geometry, or encoded references
            mismatch.
    """
    validate_canvas(width, height)
    minimum = align_num_frames(MIN_DURATION)
    maximum = align_num_frames(MAX_DURATION)
    if type(num_frames) is not int or not minimum <= num_frames <= maximum:
        raise ValueError(
            f"num_frames must be an aligned H3 frame count from {minimum} "
            f"through {maximum}"
        )
    num_latent_frames = video_latent_num_frames(num_frames)
    latent_height = height // VAE_SPATIAL_COMPRESSION_RATIO
    latent_width = width // VAE_SPATIAL_COMPRESSION_RATIO
    num_audio_latents = audio_latent_num_frames(num_frames)
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[0] != 1:
        raise ValueError("prompt_embeds must have shape [1, text_rows, hidden]")
    if (
        text_token_tags.ndim != 1
        or prompt_embeds.shape[1] != text_token_tags.numel()
    ):
        raise ValueError("text_token_tags must identify every prompt-embedding row")

    layout = build_ref2va_packed_layout(
        text_token_tags,
        references,
        encoded,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
    )

    target_device = torch.device(device or prompt_embeds.device)
    condition_rows: list[Tensor] = []
    noise_level = torch.tensor(
        KEYFRAME_NOISE_AUG, dtype=torch.float32, device=target_device
    )
    for condition in encoded.video:
        clean = condition.to(target_device, torch.float32)
        noise = _randn_tensor(
            tuple(clean.shape), generator=generator, device=target_device
        )
        noised = noise_level * clean + (1.0 - noise_level) * noise
        condition_rows.append(patchify_video_latents(noised))

    expected_video_shape = (
        1,
        VAE_LATENT_CHANNELS,
        num_latent_frames,
        latent_height,
        latent_width,
    )
    if video_noise is None:
        video_noise = _randn_tensor(
            expected_video_shape, generator=generator, device=target_device
        )
    elif tuple(video_noise.shape) != expected_video_shape:
        raise ValueError(f"video_noise must have shape {expected_video_shape}")
    target_video_rows = patchify_video_latents(
        video_noise.to(target_device, torch.float32)
    )
    packed_video = torch.cat(condition_rows + [target_video_rows])

    expected_audio_shape = (
        AUDIO_CHANNELS,
        AUDIO_LATENT_CHANNELS,
        num_audio_latents,
    )
    if audio_noise is None:
        target_audio_rows = _randn_tensor(
            (num_audio_latents * AUDIO_CHANNELS, AUDIO_LATENT_CHANNELS),
            generator=generator,
            device=target_device,
        )
    else:
        if tuple(audio_noise.shape) != expected_audio_shape:
            raise ValueError(f"audio_noise must have shape {expected_audio_shape}")
        target_audio_rows = (
            audio_noise.to(target_device, torch.float32)
            .permute(0, 2, 1)
            .reshape(-1, AUDIO_LATENT_CHANNELS)
        )
    packed_audio = torch.cat(
        [rows.to(target_device, torch.float32) for rows in encoded.audio]
        + [target_audio_rows]
    )

    if packed_video.shape[0] != (
        layout.num_condition_video_rows + target_video_rows.shape[0]
    ):
        raise ValueError("packed reference video rows do not match the layout")
    if packed_audio.shape[0] != (
        layout.num_condition_audio_rows + target_audio_rows.shape[0]
    ):
        raise ValueError("packed reference audio rows do not match the layout")
    return MiniMaxH3DenoiseState(
        latents=packed_video,
        audio_latents=packed_audio,
        prompt_embeds=prompt_embeds.to(target_device),
        position_ids=layout.position_ids.to(target_device),
        token_tags=layout.token_tags.to(target_device),
        video_indices=layout.video_indices.to(target_device),
        audio_indices=layout.audio_indices.to(target_device),
        text_indices=layout.text_indices.to(target_device),
        num_condition_video_rows=layout.num_condition_video_rows,
        num_condition_audio_rows=layout.num_condition_audio_rows,
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        audio_channels=AUDIO_CHANNELS,
    )


__all__ = [
    "MiniMaxH3PackedLayout",
    "audio_latent_num_frames",
    "build_packed_layout",
    "build_ref2va_packed_layout",
    "patchify_video_latents",
    "prepare_denoise_state",
    "prepare_ref2va_denoise_state",
    "resolve_canvas_size",
    "video_latent_num_frames",
]
