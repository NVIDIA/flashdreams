# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax and HuggingFace Teams
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax H3 presentation packing and staged native conditioning."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image, ImageOps
from flashdreams.infra.acceleration.encoder_lifecycle import (
    run_one_shot_stage,
    collect_and_release_cuda_memory,
)
from .references import MiniMaxH3Reference, MiniMaxH3ReferenceSpec, load_references

MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4


def _stage(factory: Callable, operation: Callable) -> Any:
    holder = []

    def compute():
        holder.append(factory())
        return operation(holder[0])

    def release():
        holder.clear()
        collect_and_release_cuda_memory()

    return run_one_shot_stage(compute, release=release)


def normalize_references(
    references: list[MiniMaxH3Reference], num_frames: int
) -> list[MiniMaxH3Reference]:
    """Normalize reference media at the released image, video and audio rates."""
    normalized = []
    for ref in references:
        audio = ref.audio
        if audio is not None:
            if ref.sample_rate != 32000:
                raise ValueError("Reference loader must resample audio to 32000 Hz")
            audio = audio.float()[:, : int(num_frames / 24 * 32000)]
            if audio.shape[0] == 1:
                audio = audio.expand(2, -1).contiguous()
            if audio.shape[0] != 2 or not audio.shape[1]:
                raise ValueError(
                    "Reference audio must contain nonempty mono or stereo samples"
                )
        if ref.kind == "image":
            image = ref.image
            width, height = image.size
            if not 1 / 4 <= width / height <= 4:
                raise ValueError(
                    "Reference image aspect ratio must be between 1:4 and 4:1"
                )
            scale = 2048 / min(width, height)
            size = (
                max(32, round(width * scale / 32) * 32),
                max(32, round(height * scale / 32) * 32),
            )
            normalized.append(
                MiniMaxH3Reference(
                    kind="image", image=image.resize(size, Image.Resampling.LANCZOS)
                )
            )
        elif ref.kind == "video":
            normalized.append(
                MiniMaxH3Reference(
                    kind="video",
                    frames=_normalize_video_condition(
                        ref.frames, ref.fps, num_frames, 32, 768, 768 * 1344, 24
                    ),
                    fps=24,
                    audio=audio,
                    sample_rate=32000 if audio is not None else None,
                )
            )
        else:
            normalized.append(
                MiniMaxH3Reference(kind="audio", audio=audio, sample_rate=32000)
            )
    return normalized


def prepare_keyframes(
    image_path: Path | None, last_image_path: Path | None, width: int, height: int
) -> tuple[list[Image.Image], tuple[str, ...]]:
    """Stretch the geometry anchor and cover-crop its optional follower."""
    frames, anchors = [], []
    for anchor, path in (("first", image_path), ("last", last_image_path)):
        if path is None:
            continue
        with Image.open(path) as source:
            frame = ImageOps.exif_transpose(source).convert("RGB")
        if not frames:
            frame = frame.resize((width, height), Image.Resampling.LANCZOS)
        else:
            scale = max(width / frame.width, height / frame.height)
            size = (
                max(width, round(frame.width * scale)),
                max(height, round(frame.height * scale)),
            )
            left, top = (size[0] - width) // 2, (size[1] - height) // 2
            frame = frame.resize(size, Image.Resampling.LANCZOS).crop(
                (left, top, left + width, top + height)
            )
        frames.append(frame)
        anchors.append(anchor)
    return frames, tuple(anchors)


def encode_visual_condition(encoder: Any, pixels: torch.Tensor) -> torch.Tensor:
    """Sample seed-42 visual conditioning and apply the released fp16 rounding."""
    mean = pixels.new_tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(
        1, 3, 1, 1, 1
    )
    std = pixels.new_tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(
        1, 3, 1, 1, 1
    )
    pixels = (pixels.float() / 255 - mean) / std
    latents = encoder.sample(pixels, generator=torch.Generator().manual_seed(42))
    latents = latents.half().float().cpu()
    mean = torch.tensor(encoder.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(encoder.config.latents_std).view(1, -1, 1, 1, 1)
    return (latents - mean) / std


def condition_request(
    *,
    prompt: str,
    workflow: str,
    width: int,
    height: int,
    num_frames: int,
    qwen_encoder_factory: Callable,
    image_path: Path | None = None,
    last_image_path: Path | None = None,
    references: tuple[MiniMaxH3ReferenceSpec, ...] = (),
    video_encoder_factory: Callable | None = None,
    audio_encoder_factory: Callable | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Prepare one request with only one heavyweight encoder resident at a time."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be nonempty text")
    if min(width, height) <= 0 or width % 32 or height % 32:
        raise ValueError("Canvas dimensions must be positive multiples of 32")
    if not 1 / 4 <= width / height <= 4:
        raise ValueError("Canvas aspect ratio must be between 1:4 and 4:1")
    if num_frames % 17 != 5 or not 5 <= num_frames / 24 <= 15:
        raise ValueError(
            "Frame count must be 17*n+5 and duration between 5 and 15 seconds"
        )
    if workflow not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported H3 workflow: {workflow}")
    if workflow != "fl2va" and (image_path is not None or last_image_path is not None):
        raise ValueError("Keyframes require fl2va")
    if workflow != "ref2va" and references:
        raise ValueError("Ordered references require ref2va")
    keyframes, anchors = prepare_keyframes(image_path, last_image_path, width, height)
    if workflow == "fl2va" and not keyframes:
        raise ValueError("fl2va requires a first or last keyframe")
    refs = (
        normalize_references(load_references(references), num_frames)
        if workflow == "ref2va"
        else []
    )
    visual_refs = (
        [MiniMaxH3Reference(kind="image", image=image) for image in keyframes]
        if keyframes
        else refs
    )

    def encode_video(encoder):
        conditions = []
        for reference in visual_refs:
            if reference.kind == "image":
                pixels = torch.from_numpy(np.array(reference.image)).permute(2, 0, 1)[
                    None, :, None
                ]
            elif reference.kind == "video":
                count = max(1, (len(reference.frames) - 5) // 17) * 17 + 5
                pixels = torch.from_numpy(reference.frames[:count].copy()).permute(
                    3, 0, 1, 2
                )[None]
            else:
                continue
            conditions.append(encode_visual_condition(encoder, pixels.to(device)))
        return conditions

    has_visual = any(ref.kind in {"image", "video"} for ref in visual_refs)
    if has_visual and video_encoder_factory is None:
        raise ValueError("Visual conditioning requires a video encoder factory")
    conditions = _stage(video_encoder_factory, encode_video) if has_visual else []

    def encode_audio(encoder):
        mean = torch.tensor(encoder.config.latents_mean).view(1, 1, -1)
        std = torch.tensor(encoder.config.latents_std).view(1, 1, -1)
        return [
            (
                (
                    encoder.encode(ref.audio.to(device)[:, None])
                    .float()
                    .cpu()
                    .transpose(1, 2)
                    - mean
                )
                / std
            ).reshape(-1, 32)
            for ref in refs
            if ref.has_audio
        ]

    has_audio = any(ref.has_audio for ref in refs)
    if has_audio and audio_encoder_factory is None:
        raise ValueError("Audio references require an audio encoder factory")
    audio_conditions = _stage(audio_encoder_factory, encode_audio) if has_audio else []

    def encode_text(encoder):
        vision, image_counts, video_counts, timestamps = _gather_vision_features(
            encoder.processor, visual_refs, 24
        )
        ids, tags = _build_presentation(
            encoder.tokenizer,
            prompt,
            visual_refs,
            image_counts,
            video_counts,
            timestamps,
        )
        embeddings = encoder({"token_ids": ids, "vision_inputs": vision})
        return {
            "prompt_embeds": embeddings,
            "text_token_tags": torch.tensor(tags, dtype=torch.long),
        }

    return {
        **_stage(qwen_encoder_factory, encode_text),
        "condition_latents": conditions,
        "audio_condition_latents": audio_conditions,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "keyframe_anchors": anchors,
        "normalized_references": refs,
    }


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int,
    short_edge: int,
    max_pixels: int,
    min_aspect_ratio: float = MINIMAX_H3_MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MINIMAX_H3_MAX_ASPECT_RATIO,
) -> tuple[int, int]:
    """Resolve a display aspect ratio into a MiniMax-H3 canvas."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(
            f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}."
        )

    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:{1 / min_aspect_ratio:g} to {max_aspect_ratio:g}:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio

    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale

    multiple = canvas_multiple
    return max(multiple, round(height / multiple) * multiple), max(
        multiple, round(width / multiple) * multiple
    )


def _normalize_video_condition(
    frames,
    fps: float,
    num_frames: int,
    canvas_multiple: int,
    canvas_short_edge: int,
    canvas_max_pixels: int,
    target_fps: float,
) -> np.ndarray:
    """Normalize a video reference's frames: any accepted layout, onto `uint8` at 24 fps, truncated to the generated"""
    # Any accepted layout onto `uint8` THWC. A `torch.Tensor` is channels-first, as everywhere else in
    # diffusers, and a `np.ndarray` channels-last; floating point values are read over `[0, 1]`.
    if isinstance(frames, list):
        frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    if isinstance(frames, torch.Tensor):
        frames = frames.movedim(-3, -1).cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"A reference video must be `(num_frames, height, width, 3)` RGB frames, got {tuple(frames.shape)}."
        )

    # Onto MiniMax-H3's 24 fps grid: every frame is held until the slot of the next one, and the last one until
    # the slot the stream's end rounds to.
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"A reference video must have a positive frame rate, got {fps}."
        )
    if fps != target_fps:
        scale = target_fps / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(
            frames,
            np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)),
            axis=0,
        )

    # Truncated to the generated frame count and put on the canvas of its *own* aspect ratio — the same rule the
    # target canvas follows, unlike an image reference.
    frames = frames[:num_frames]
    if not len(frames):
        raise ValueError("Reference video is too short to contain a frame at 24 fps")
    height, width = resolve_canvas_size(
        frames.shape[2],
        frames.shape[1],
        canvas_multiple,
        canvas_short_edge,
        canvas_max_pixels,
    )
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)
            )
            for frame in frames
        ]
    )


def _sample_video_condition_frames(
    frames: np.ndarray, fps: float, sample_fps: float, temporal_patch: int
) -> tuple[list[np.ndarray], list[float]]:
    """Sample the frames the conditioner sees from a normalized reference video, and label their vision blocks."""
    stride = fps / sample_fps
    indices, cursor = [], 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"A reference video is read at {sample_fps:g} fps and its sampled frames are merged in groups of "
            f"{temporal_patch}, so it must run at least {minimum} frames at {fps:g} fps "
            f"({minimum / fps:.2g} seconds), got {frames.shape[0]}."
        )

    timestamps = [index / sample_fps for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    block_timestamps = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return [frames[index] for index in indices], block_timestamps


def _gather_vision_features(
    processor, references: list[MiniMaxH3Reference], fps: float
) -> tuple[dict, list[int], list[int], list[list[float]]]:
    """Run the references' pixels through the conditioner's processors, batched per modality."""
    merge_size = processor.image_processor.merge_size**2
    vision_inputs = {}

    image_token_counts = []
    images = [reference.image for reference in references if reference.kind == "image"]
    if images:
        image_features = processor.image_processor(images=images, return_tensors="pt")
        vision_inputs["pixel_values"] = image_features["pixel_values"]
        vision_inputs["image_grid_thw"] = image_features["image_grid_thw"]
        image_token_counts = [
            int(grid.prod()) // merge_size for grid in image_features["image_grid_thw"]
        ]

    video_block_token_counts, video_block_timestamps = [], []
    videos = [reference for reference in references if reference.kind == "video"]
    if videos:
        temporal_patch = processor.video_processor.temporal_patch_size
        sampled = [
            _sample_video_condition_frames(reference.frames, fps, 2.0, temporal_patch)
            for reference in videos
        ]
        video_block_timestamps = [timestamps for _, timestamps in sampled]
        video_features = processor.video_processor(
            videos=[np.stack(frames) for frames, _ in sampled],
            do_sample_frames=False,
            return_tensors="pt",
        )
        vision_inputs["pixel_values_videos"] = video_features["pixel_values_videos"]
        vision_inputs["video_grid_thw"] = video_features["video_grid_thw"]
        video_block_token_counts = [
            int(grid[1]) * int(grid[2]) // merge_size
            for grid in video_features["video_grid_thw"]
        ]
        for timestamps, grid in zip(
            video_block_timestamps, video_features["video_grid_thw"]
        ):
            if int(grid[0]) != len(timestamps):
                raise ValueError(
                    f"The processor merged a reference video into {int(grid[0])} vision blocks, but MiniMax-H3 "
                    f"labels {len(timestamps)} of them."
                )

    return (
        vision_inputs,
        image_token_counts,
        video_block_token_counts,
        video_block_timestamps,
    )


def _build_presentation(
    tokenizer,
    prompt: str,
    references: list[MiniMaxH3Reference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
    video_block_timestamps: list[list[float]],
    text_tag: int = 1,
    video_tag: int = 0,
) -> tuple[list[int], list[int]]:
    """Tokenize MiniMax-H3's presentation of a `ref2va` request."""

    def text(value: str) -> tuple[list[int], list[int]]:
        token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        return token_ids, [text_tag] * len(token_ids)

    def vision(pad_token: str, num_tokens: int) -> tuple[list[int], list[int]]:
        token_ids = (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )
        return token_ids, [video_tag] * len(token_ids)

    token_ids, token_tags = [], []

    def emit(segment: tuple[list[int], list[int]]) -> None:
        token_ids.extend(segment[0])
        token_tags.extend(segment[1])

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit(text(f"<Audio {counts['audio']}>: "))
        if reference.kind == "image":
            counts["image"] += 1
            emit(text(f"<Picture {counts['image']}>: "))
            emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
        elif reference.kind == "video":
            counts["video"] += 1
            emit(text(f"<Video {counts['video']}>: "))
            for timestamp in video_block_timestamps[counts["video"] - 1]:
                # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
                emit(text(f"<{timestamp:.1f} seconds>"))
                emit(
                    vision(
                        "<|video_pad|>", video_block_token_counts[counts["video"] - 1]
                    )
                )
    emit(text(prompt))
    return token_ids, token_tags
