# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax and HuggingFace Teams
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax H3 packed row geometry and deterministic initial noise."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from .references import MiniMaxH3Reference


@dataclass
class MiniMaxH3DenoiseState:
    """Packed conditioning, seeded noise and geometry for one request."""

    latents: torch.Tensor
    """Video rows, immutable conditioning prefix first."""
    audio_latents: torch.Tensor
    """Audio rows, immutable conditioning prefix first."""
    prompt_embeds: torch.Tensor
    """Raw Qwen hidden states."""
    position_ids: torch.Tensor
    """Float64 rotary coordinates of every sequence row."""
    token_tags: torch.Tensor
    """Per-row AdaLN modality identifiers."""
    video_indices: torch.Tensor
    """Sequence positions of video rows."""
    audio_indices: torch.Tensor
    """Sequence positions of audio rows."""
    text_indices: torch.Tensor
    """Sequence positions of Qwen conditioning rows."""
    num_condition_video_rows: int
    """Video prefix length held fixed while denoising."""
    num_condition_audio_rows: int
    """Audio prefix length held fixed while denoising."""
    num_latent_frames: int
    """Target video latent frame count."""
    latent_height: int
    """Target latent canvas height."""
    latent_width: int
    """Target latent canvas width."""


def prepare_denoise_state(
    conditioned: dict, seed: int, workflow: str, device: str | torch.device = "cpu"
) -> MiniMaxH3DenoiseState:
    """Pack conditions before drawing target video and then audio noise."""
    frames = conditioned["num_frames"]
    if frames % 17 != 5:
        raise ValueError("Video frame count must be 17*n+5")
    height, width = conditioned["height"] // 16, conditioned["width"] // 16
    latent_frames = (frames - 5) // 17 * 5 + 2
    audio_frames = round(frames / 24 * 40)
    conditions = conditioned.get("condition_latents", [])
    audio_conditions = conditioned.get("audio_condition_latents", [])
    tags = conditioned["text_token_tags"].cpu()
    if workflow == "ref2va":
        layout = build_ref2va_packed_sequence(
            tags,
            conditioned["normalized_references"],
            conditions,
            audio_conditions,
            latent_frames,
            height,
            width,
            audio_frames,
            (1, 2, 2),
            2,
            2,
            0,
        )
    elif workflow in {"t2va", "fl2va"}:
        layout = build_packed_sequence(
            tags,
            latent_frames,
            height,
            width,
            audio_frames,
            (1, 2, 2),
            2,
            2,
            0,
            conditioned.get("keyframe_anchors", ()),
        )
    else:
        raise ValueError(f"Unsupported H3 workflow: {workflow}")
    position_ids, token_tags, video_indices, audio_indices, text_indices, nv, na = (
        layout
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    packed = []
    # CPU RNG and draw boundaries match the released request generator exactly.
    for condition in conditions:
        condition = condition.to(device=device, dtype=torch.float32)
        noise = torch.randn(
            condition.shape, generator=generator, dtype=torch.float32
        ).to(device)
        timestep = torch.tensor(0.999, dtype=torch.float32, device=device)
        noised = timestep * condition + (1 - timestep) * noise
        packed.append(patchify_video_latents(noised, (1, 2, 2)))
    if sum(value.shape[0] for value in packed) != nv:
        raise ValueError("Conditioning rows do not match the requested canvas")
    video = torch.randn(
        (1, 24, latent_frames, height, width), generator=generator, dtype=torch.float32
    ).to(device)
    video = patchify_video_latents(video, (1, 2, 2))
    audio = torch.randn(
        (audio_frames * 2, 32), generator=generator, dtype=torch.float32
    ).to(device)
    if sum(value.shape[0] for value in audio_conditions) != na:
        raise ValueError("Audio conditioning rows do not match the reference layout")
    return MiniMaxH3DenoiseState(
        latents=torch.cat([*packed, video]),
        audio_latents=torch.cat(
            [*(value.to(device) for value in audio_conditions), audio]
        ),
        prompt_embeds=conditioned["prompt_embeds"].to(device),
        position_ids=position_ids.to(device),
        token_tags=token_tags.to(device),
        video_indices=video_indices.to(device),
        audio_indices=audio_indices.to(device),
        text_indices=text_indices.to(device),
        num_condition_video_rows=nv,
        num_condition_audio_rows=na,
        num_latent_frames=latent_frames,
        latent_height=height,
        latent_width=width,
    )


_ROPE_FRAME_RESCALE = 5.0 / 3.0
"""Rotary units per pixel frame."""
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
"""Released video VAE's nonuniform temporal grouping."""
_ROPE_SPATIAL_SCALE = 32


def patchify_video_latents(
    latents: torch.Tensor, patch_size: tuple[int, int, int]
) -> torch.Tensor:
    """Pack video latents into transformer rows."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}."
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
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """One aspect-normalized spatial rotary axis: `dim // patch` coordinates centred on the unit interval, scaled up by"""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # Built with numpy: `np.linspace(..., endpoint=False)` is `start + arange(num) * (stop - start) / num`, which is
    # not what `torch.linspace` computes, and the float64 grid has to be reproduced exactly.
    grid = (
        np.linspace(left, left + ratio, dim // patch, endpoint=False)
        * _ROPE_SPATIAL_SCALE
    )
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """The rotary time of every latent frame, starting at `origin`. Spacing is non-uniform: `5/3 * (1, 4, 4, 4, 4)`."""
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
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
    audio_channels: int,
) -> None:
    """Place one channel-major audio block."""
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(audio_channels)
    position_ids[rows, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full(
                (num_audio_latents,), float(width_grid[-1]), dtype=torch.float64
            ),
        ]
    )


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    audio_channels: int,
    audio_tag: int,
    video_tag: int,
    keyframe_anchors: tuple[str, ...] = (),
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int
]:
    """Build the `[text | keyframe conditions | target audio | target video]` layout used by the `t2va` and `fl2va`"""
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = (
        num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
    )

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) grid. Text rows sit on the time axis at their row index, and the media rows continue the time
    # axis from there, so text length shifts the whole media clock.
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
            # The rotary time the generated frames span, summed by numpy's pairwise summation because that is
            # how the reference computes this anchor. The soundtrack span in the `ref2va` layout sums the same
            # series sequentially, and the two orders differ in the last ulp from 16 latent frames onwards, so
            # the port keeps both — one per call site.
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= (
                    _ROPE_FRAMES_PER_LATENT[offset]
                )
            anchor_time = (
                float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(
                f"A keyframe anchor must be 'first' or 'last', got {anchor!r}."
            )
        rows = slice(
            condition_start + index * rows_per_frame,
            condition_start + (index + 1) * rows_per_frame,
        )
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are channel-major and share the video's rotary clock: one unit per latent at 40 latents/s equals
    # 24 fps * 5/3. They carry no height coordinate and are pinned to the two extremes of the width grid.
    audio_time = float(num_text_tokens) + torch.arange(
        num_audio_latents, dtype=torch.float64
    )
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full(
                (num_audio_rows - num_audio_latents,),
                float(width_grid[-1]),
                dtype=torch.float64,
            ),
        ]
    )

    video_position_ids = torch.empty(
        num_latent_frames, rows_per_frame, 3, dtype=torch.float64
    )
    video_position_ids[:, :, 0] = _temporal_position_grid(
        num_latent_frames, float(num_text_tokens)
    )[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # 2. Row indices and modality tags.
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
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag

    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_rows,
        0,
    )


def build_ref2va_packed_sequence(
    text_token_tags: torch.Tensor,
    references: list[MiniMaxH3Reference],
    condition_latents: list[torch.Tensor],
    audio_condition_latents: list[torch.Tensor],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    audio_channels: int,
    audio_tag: int,
    video_tag: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int
]:
    """Build the `[text | reference blocks | target audio | target video]` layout of the `ref2va` task."""
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_target_video_rows = (
        num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    )
    num_target_audio_rows = num_audio_latents * audio_channels

    # The geometry of every reference block is the shape of what the encoder produced for it, so the two can never
    # disagree. Both lists are in packed order but skip the references they do not apply to, so they are consumed as
    # iterators alongside the reference list rather than indexed by it.
    visual_geometry = iter(tuple(latents.shape[2:5]) for latents in condition_latents)
    audio_row_counts = iter(rows.shape[0] for rows in audio_condition_latents)
    num_reference_video_rows = sum(
        frames * (height // patch_h) * (width // patch_w)
        for frames, height, width in (
            tuple(latents.shape[2:5]) for latents in condition_latents
        )
    )
    num_reference_audio_rows = sum(rows.shape[0] for rows in audio_condition_latents)
    sequence_length = (
        num_text_tokens
        + num_reference_video_rows
        + num_reference_audio_rows
        + num_target_audio_rows
        + num_target_video_rows
    )

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(
        num_text_tokens, dtype=torch.float64
    )
    target_frame_grid, target_width_grid = _frame_position_grid(
        latent_height, latent_width, patch_h, patch_w
    )

    # Reference blocks, in request order. `rotary_time` is the shared audio/video clock: it starts where the text
    # rows end and every block pushes it forward by the time that block occupies.
    video_indices, audio_indices = [], []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for reference in references:
        if reference.kind == "image":
            num_latent_frames_, reference_height, reference_width = next(
                visual_geometry
            )
            num_video_rows = (
                num_latent_frames_
                * (reference_height // patch_h)
                * (reference_width // patch_w)
            )
            rows = slice(cursor, cursor + num_video_rows)
            cursor = rows.stop
            video_indices.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(
                reference_height, reference_width, patch_h, patch_w
            )
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid
            # An image is a single frame and takes a single integer rotary slot, not a latent frame's 5/3 units.
            rotary_time += 1.0
        elif reference.kind == "audio":
            num_audio_rows = next(audio_row_counts)
            reference_audio_latents = num_audio_rows // audio_channels
            rows = slice(cursor, cursor + num_audio_rows)
            cursor = rows.stop
            audio_indices.append(torch.arange(rows.start, rows.stop))
            _fill_audio_positions(
                position_ids,
                rows,
                reference_audio_latents,
                rotary_time,
                target_width_grid,
                audio_channels,
            )
            rotary_time += float(reference_audio_latents)
        elif reference.kind == "video":
            # A video reference's soundtrack rows are packed immediately before its video rows and share their
            # origin, so the two are rotary-aligned exactly as the generated audio and video are.
            num_audio_rows = next(audio_row_counts) if reference.has_audio else 0
            reference_audio_latents = num_audio_rows // audio_channels
            num_latent_frames_, reference_height, reference_width = next(
                visual_geometry
            )
            num_video_rows = (
                num_latent_frames_
                * (reference_height // patch_h)
                * (reference_width // patch_w)
            )
            audio_rows = slice(cursor, cursor + num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
            cursor = video_rows.stop
            audio_indices.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_indices.append(torch.arange(video_rows.start, video_rows.stop))

            frame_grid, width_grid = _frame_position_grid(
                reference_height, reference_width, patch_h, patch_w
            )
            _fill_audio_positions(
                position_ids,
                audio_rows,
                reference_audio_latents,
                rotary_time,
                width_grid,
                audio_channels,
            )
            frame_time = _temporal_position_grid(num_latent_frames_, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(
                frame_grid.shape[0]
            )
            position_ids[video_rows, 1:] = frame_grid.repeat(num_latent_frames_, 1)
            # The rotary time this reference advances the clock by, summed sequentially in float64. That is *not*
            # how `_temporal_position_span_pairwise` sums the same series — it reproduces a numpy pairwise sum,
            # and the two orders differ in the last ulp from 16 latent frames onwards. The reference
            # implementation keeps both, one per call site, so the port has to as well.
            video_span = sum(
                _ROPE_FRAME_RESCALE
                * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
                for index in range(num_latent_frames_)
            )
            rotary_time += max(float(reference_audio_latents), video_span)
        else:
            raise ValueError(
                f"A reference must be an 'image', a 'video' or an 'audio', got {reference.kind!r}."
            )

    # The generated rows. Target audio and target video share the origin the reference blocks left behind.
    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(
        position_ids,
        slice(audio_start, video_start),
        num_audio_latents,
        rotary_time,
        target_width_grid,
        audio_channels,
    )
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(
        target_frame_grid.shape[0]
    )
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(
        video_indices + [torch.arange(video_start, sequence_length)]
    )
    audio_indices = torch.cat(audio_indices + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag

    return (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_reference_video_rows,
        num_reference_audio_rows,
    )
