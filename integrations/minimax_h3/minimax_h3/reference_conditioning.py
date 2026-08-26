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

"""In-memory MiniMax H3 REF2VA references and deterministic normalization.

Modified from the Apache-2.0 H3 reference setup in Hugging Face Diffusers
commit ``175fe6b2419a01db9c2ceabd01ec37d2c0305fc2``. File and URL decoding
belongs to the V2 application boundary; this module accepts decoded media only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from minimax_h3.constants import (
    AUDIO_CHANNELS,
    AUDIO_LATENT_CHANNELS,
    AUDIO_SAMPLE_RATE,
    CANVAS_MAX_PIXELS,
    CANVAS_MULTIPLE,
    CANVAS_SHORT_EDGE,
    FPS,
    FRAME_CHUNK,
    FRAME_REMAINDER,
    MAX_DURATION,
    MIN_DURATION,
    VAE_LATENT_CHANNELS,
    align_num_frames,
)

REFERENCE_IMAGE_SHORT_EDGE = 2048
"""Released short-edge size for high-detail image references."""

MAX_IMAGE_REFERENCES = 9
MAX_VIDEO_REFERENCES = 3
MAX_AUDIO_REFERENCES = 3
MAX_REFERENCES = 12


@dataclass(frozen=True, slots=True)
class MiniMaxH3Reference:
    """Base type for one decoded REF2VA reference."""

    @property
    def kind(self) -> str:
        """Return the reference modality name."""
        raise NotImplementedError

    @property
    def has_audio(self) -> bool:
        """Return whether the reference contributes soundtrack rows."""
        return False


@dataclass(frozen=True, slots=True)
class MiniMaxH3ImageReference(MiniMaxH3Reference):
    """One decoded RGB subject, style, or scene reference."""

    image: Image.Image | np.ndarray | Tensor

    @property
    def kind(self) -> str:
        """Return the image modality name."""
        return "image"


@dataclass(frozen=True, slots=True)
class MiniMaxH3VideoReference(MiniMaxH3Reference):
    """Decoded RGB frames with their rate and optional soundtrack."""

    frames: list[Image.Image] | np.ndarray | Tensor
    fps: float | None = None
    audio: Tensor | np.ndarray | None = None
    sample_rate: int | None = None

    @property
    def kind(self) -> str:
        """Return the video modality name."""
        return "video"

    @property
    def has_audio(self) -> bool:
        """Return whether this reference contributes soundtrack rows."""
        return self.audio is not None


@dataclass(frozen=True, slots=True)
class MiniMaxH3AudioReference(MiniMaxH3Reference):
    """One decoded mono or stereo waveform reference."""

    audio: Tensor | np.ndarray
    sample_rate: int | None = None

    @property
    def kind(self) -> str:
        """Return the audio modality name."""
        return "audio"

    @property
    def has_audio(self) -> bool:
        """Return that audio references always contribute audio rows."""
        return True


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3EncodedReferences:
    """Video and audio conditioning rows encoded in semantic reference order."""

    video: tuple[Tensor, ...]
    """One normalized video-VAE latent tensor per visual reference."""

    audio: tuple[Tensor, ...]
    """One normalized channel-major row matrix per audio-bearing reference."""


def _validate_num_frames(num_frames: int) -> None:
    """Validate a final H3 frame count, including the aligned 15 s result."""
    minimum = align_num_frames(MIN_DURATION)
    maximum = align_num_frames(MAX_DURATION)
    if type(num_frames) is not int or not minimum <= num_frames <= maximum:
        raise ValueError(
            f"num_frames must be an aligned H3 frame count from {minimum} "
            f"through {maximum}"
        )
    if num_frames % 17 != 5:
        raise ValueError("num_frames must be aligned to the H3 frame grid")


def _normalize_image(
    image: Image.Image | np.ndarray | Tensor,
    *,
    short_edge: int,
    multiple: int,
) -> Image.Image:
    """Convert one accepted image layout to resized RGB pixels."""
    if isinstance(image, Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                "A reference image tensor must have shape [3, height, width], "
                f"got {tuple(image.shape)}."
            )
        image = image.detach().movedim(0, -1).cpu().numpy()
    if isinstance(image, np.ndarray):
        image_array = cast(np.ndarray, image)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(
                "A reference image array must have shape [height, width, 3], "
                f"got {tuple(image_array.shape)}."
            )
        if image_array.dtype != np.uint8:
            if not np.isfinite(image_array).all():
                raise ValueError("A reference image must contain finite pixels.")
            image_array = (
                (image_array * 255.0).round().clip(0, 255).astype(np.uint8)
            )
        image = Image.fromarray(image_array, mode="RGB")
    if not isinstance(image, Image.Image):
        raise ValueError(
            "A reference image must be a PIL image, NumPy array, or Torch tensor."
        )
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(
            f"A reference image must have a positive size, got {image.size}."
        )
    if width > 4 * height or height > 4 * width:
        raise ValueError(
            f"A reference image must be within 1:4 and 4:1, got {width}x{height}."
        )
    scale = short_edge / min(width, height)
    target_height = max(multiple, round(height * scale / multiple) * multiple)
    target_width = max(multiple, round(width * scale / multiple) * multiple)
    if image.size != (target_width, target_height):
        image = image.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )
    return image


def _video_to_uint8(frames: list[Image.Image] | np.ndarray | Tensor) -> np.ndarray:
    """Convert accepted reference-video layouts to contiguous uint8 THWC."""
    if isinstance(frames, list):
        if not frames:
            raise ValueError("A reference video must contain at least one frame.")
        frames = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    elif isinstance(frames, Tensor):
        frames = frames.detach().movedim(-3, -1).cpu().numpy()
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        if not np.isfinite(frames).all():
            raise ValueError("A reference video must contain finite pixels.")
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[0] == 0 or frames.shape[3] != 3:
        raise ValueError(
            "A reference video must have shape [frames, height, width, 3], "
            f"got {tuple(frames.shape)}."
        )
    return np.ascontiguousarray(frames)


def _normalize_video(
    frames: list[Image.Image] | np.ndarray | Tensor,
    *,
    fps: float,
    num_frames: int,
    target_fps: float,
    canvas_short_edge: int,
    canvas_max_pixels: int,
    multiple: int,
) -> np.ndarray:
    """Resample decoded video onto H3's whole-frame clock and canvas."""
    from minimax_h3.conditioning import resolve_canvas_size

    frames = _video_to_uint8(frames)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"A reference video must have a positive frame rate, got {fps}."
        )
    if fps != target_fps:
        scale = target_fps / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(
            np.int64
        )
        end = math.floor(frames.shape[0] * scale + 0.5)
        frames = np.repeat(frames, np.diff(slots, append=end), axis=0)
    frames = frames[:num_frames]
    if frames.shape[0] == 0:
        raise ValueError("A reference video must occupy at least one H3 frame.")
    height, width = resolve_canvas_size(
        frames.shape[2],
        frames.shape[1],
        short_edge=canvas_short_edge,
        max_pixels=canvas_max_pixels,
        multiple=multiple,
    )
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            )
            for frame in frames
        ]
    )


def _normalize_audio(
    waveform: Tensor | np.ndarray,
    *,
    sample_rate: int,
    target_sample_rate: int,
    max_samples: int,
) -> Tensor:
    """Validate, truncate, and upmix already-resampled reference audio."""
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError(
            "A reference soundtrack needs a positive sample rate, got "
            f"{sample_rate}."
        )
    if sample_rate != target_sample_rate:
        raise ValueError(
            f"Reference audio must be decoded at {target_sample_rate} Hz by the "
            f"V2 media loader, got {sample_rate} Hz."
        )
    waveform = torch.as_tensor(waveform).detach().to(device="cpu", dtype=torch.float32)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(
            "A reference soundtrack must have shape [1|2, samples], got "
            f"{tuple(waveform.shape)}."
        )
    if waveform.shape[1] == 0:
        raise ValueError("A reference soundtrack must contain at least one sample.")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("A reference soundtrack must contain finite samples.")
    waveform = waveform[:, :max_samples]
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1).contiguous()
    return waveform.contiguous()


def normalize_references(
    references: list[MiniMaxH3Reference],
    *,
    num_frames: int,
    target_fps: float = FPS,
    target_sample_rate: int = AUDIO_SAMPLE_RATE,
    canvas_short_edge: int = CANVAS_SHORT_EDGE,
    canvas_max_pixels: int = CANVAS_MAX_PIXELS,
    reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    multiple: int = CANVAS_MULTIPLE,
) -> tuple[MiniMaxH3Reference, ...]:
    """Validate and normalize an ordered REF2VA reference sequence.

    The order is preserved because it determines both presentation labels and
    the shared audio/video rotary clock. Audio must already be decoded at the
    H3 audio-VAE rate by the model-neutral V2 media loader.

    Args:
        references: Ordered decoded image, video, and audio references.
        num_frames: Final aligned output frame count.
        target_fps: H3 video clock in frames per second.
        target_sample_rate: H3 audio-VAE sample rate.
        canvas_short_edge: Short-edge policy for video references.
        canvas_max_pixels: Area cap for video-reference canvases.
        reference_image_short_edge: Independent image-reference short edge.
        multiple: Spatial multiple for all normalized references.

    Returns:
        Normalized references in the original semantic order.

    Raises:
        ValueError: A limit, modality, rate, shape, or value is invalid.
    """
    _validate_num_frames(num_frames)
    if not isinstance(references, list) or not references:
        raise ValueError("REF2VA needs at least one reference.")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be positive and finite")
    integer_options = {
        "target_sample_rate": target_sample_rate,
        "canvas_short_edge": canvas_short_edge,
        "canvas_max_pixels": canvas_max_pixels,
        "reference_image_short_edge": reference_image_short_edge,
        "multiple": multiple,
    }
    if any(
        type(value) is not int or value <= 0
        for value in integer_options.values()
    ):
        raise ValueError(
            "reference normalization sizes and rates must be positive integers"
        )
    for index, reference in enumerate(references):
        if not isinstance(
            reference,
            (
                MiniMaxH3ImageReference,
                MiniMaxH3VideoReference,
                MiniMaxH3AudioReference,
            ),
        ):
            raise ValueError(
                f"references[{index}] must be an in-memory MiniMax H3 reference, "
                f"got {type(reference)}."
            )
    kinds = [reference.kind for reference in references]
    for kind, limit in (
        ("image", MAX_IMAGE_REFERENCES),
        ("video", MAX_VIDEO_REFERENCES),
        ("audio", MAX_AUDIO_REFERENCES),
    ):
        if kinds.count(kind) > limit:
            raise ValueError(
                f"MiniMax H3 accepts at most {limit} {kind} references, "
                f"got {kinds.count(kind)}."
            )
    if len(references) > MAX_REFERENCES:
        raise ValueError(
            f"MiniMax H3 accepts at most {MAX_REFERENCES} references, "
            f"got {len(references)}."
        )
    if set(kinds) == {"audio"}:
        raise ValueError(
            "An audio reference must be paired with an image or video reference."
        )

    max_samples = int(num_frames * target_sample_rate / target_fps)
    normalized: list[MiniMaxH3Reference] = []
    for reference in references:
        waveform = None
        if isinstance(reference, MiniMaxH3AudioReference) or (
            isinstance(reference, MiniMaxH3VideoReference)
            and reference.has_audio
        ):
            source_audio = reference.audio
            if source_audio is None:
                raise ValueError("An audio-bearing reference has no waveform.")
            sample_rate = (
                target_sample_rate
                if reference.sample_rate is None
                else reference.sample_rate
            )
            waveform = _normalize_audio(
                source_audio,
                sample_rate=sample_rate,
                target_sample_rate=target_sample_rate,
                max_samples=max_samples,
            )
        if isinstance(reference, MiniMaxH3ImageReference):
            normalized.append(
                MiniMaxH3ImageReference(
                    _normalize_image(
                        reference.image,
                        short_edge=reference_image_short_edge,
                        multiple=multiple,
                    )
                )
            )
        elif isinstance(reference, MiniMaxH3VideoReference):
            normalized.append(
                MiniMaxH3VideoReference(
                    frames=_normalize_video(
                        reference.frames,
                        fps=float(
                            target_fps if reference.fps is None else reference.fps
                        ),
                        num_frames=num_frames,
                        target_fps=target_fps,
                        canvas_short_edge=canvas_short_edge,
                        canvas_max_pixels=canvas_max_pixels,
                        multiple=multiple,
                    ),
                    fps=target_fps,
                    audio=waveform,
                    sample_rate=target_sample_rate if waveform is not None else None,
                )
            )
        elif isinstance(reference, MiniMaxH3AudioReference):
            if waveform is None:
                raise AssertionError("validated audio reference was not normalized")
            normalized.append(
                MiniMaxH3AudioReference(
                    audio=waveform, sample_rate=target_sample_rate
                )
            )
        else:
            raise AssertionError("validated reference type was not normalized")
    return tuple(normalized)


@torch.no_grad()
def encode_references(
    video_vae: Any,
    audio_vae: Any,
    references: list[MiniMaxH3Reference] | tuple[MiniMaxH3Reference, ...],
    *,
    device: torch.device | str,
) -> MiniMaxH3EncodedReferences:
    """Encode normalized REF2VA media with caller-owned native VAEs.

    The function moves only input tensors. Application resource policy owns
    VAE placement and offload. Visual posterior samples use the video VAE's
    independent seed-42 conditioning path; soundtrack rows take the audio
    posterior mean and remain clean throughout denoising.

    Args:
        video_vae: Loaded native :class:`MiniMaxH3VideoVAE`.
        audio_vae: Loaded native :class:`MiniMaxH3AudioVAE`.
        references: Normalized references in semantic packed order.
        device: Device on which the already-placed VAEs consume input media.

    Returns:
        Visual latent tensors and soundtrack row matrices in their respective
        filtered reference orders.

    Raises:
        ValueError: References are not normalized or an encoder result is
            malformed.
    """
    target_device = torch.device(device)
    video_latents: list[Tensor] = []
    audio_rows: list[Tensor] = []
    for reference in references:
        if isinstance(reference, MiniMaxH3ImageReference):
            if not isinstance(reference.image, Image.Image):
                raise ValueError(
                    "image references must be normalized before encoding"
                )
            pixels = (
                torch.from_numpy(np.array(reference.image, copy=True))
                .permute(2, 0, 1)[None, :, None]
                .to(device=target_device, dtype=torch.float32)
                .div_(255.0)
            )
            video_latents.append(video_vae.encode_condition_pixels(pixels))
        elif isinstance(reference, MiniMaxH3VideoReference):
            frames = reference.frames
            if not isinstance(frames, np.ndarray):
                raise ValueError(
                    "video references must be normalized before encoding"
                )
            frames_array = cast(np.ndarray, frames)
            source_frames = frames_array.shape[0]
            encoded_frames = (
                max(1, (source_frames - FRAME_REMAINDER) // FRAME_CHUNK)
                * FRAME_CHUNK
                + FRAME_REMAINDER
            )
            pixels = (
                torch.from_numpy(frames_array[:encoded_frames].copy())
                .permute(3, 0, 1, 2)[None]
                .to(device=target_device, dtype=torch.float32)
                .div_(255.0)
            )
            video_latents.append(video_vae.encode_condition_pixels(pixels))
        elif not isinstance(reference, MiniMaxH3AudioReference):
            raise ValueError(f"Unsupported MiniMax H3 reference {type(reference)}")

        if isinstance(reference, MiniMaxH3AudioReference) or (
            isinstance(reference, MiniMaxH3VideoReference)
            and reference.has_audio
        ):
            if not isinstance(reference.audio, Tensor):
                raise ValueError(
                    "audio references must be normalized before encoding"
                )
            audio_rows.append(
                audio_vae.encode_condition(reference.audio.to(target_device))
            )

    for latents in video_latents:
        if (
            latents.ndim != 5
            or latents.shape[0] != 1
            or latents.shape[1] != VAE_LATENT_CHANNELS
            or any(size <= 0 for size in latents.shape[2:])
        ):
            raise ValueError("video reference encoder returned malformed latents")
        if not latents.is_floating_point() or not bool(
            torch.isfinite(latents).all()
        ):
            raise ValueError("video reference latents must be finite floating point")
    for rows in audio_rows:
        if (
            rows.ndim != 2
            or rows.shape[0] == 0
            or rows.shape[0] % AUDIO_CHANNELS
            or rows.shape[1] != AUDIO_LATENT_CHANNELS
        ):
            raise ValueError("audio reference encoder returned malformed rows")
        if not rows.is_floating_point() or not bool(torch.isfinite(rows).all()):
            raise ValueError("audio reference rows must be finite floating point")
    return MiniMaxH3EncodedReferences(
        video=tuple(video_latents), audio=tuple(audio_rows)
    )


__all__ = [
    "MAX_AUDIO_REFERENCES",
    "MAX_IMAGE_REFERENCES",
    "MAX_REFERENCES",
    "MAX_VIDEO_REFERENCES",
    "MiniMaxH3AudioReference",
    "MiniMaxH3EncodedReferences",
    "MiniMaxH3ImageReference",
    "MiniMaxH3Reference",
    "MiniMaxH3VideoReference",
    "REFERENCE_IMAGE_SHORT_EDGE",
    "encode_references",
    "normalize_references",
]
