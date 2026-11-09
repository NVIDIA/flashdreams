# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control video in: decode, resize, scale, and encode a latent at a time.

A control video (segmentation, depth, edge or blur rendered as RGB) is resized
straight to 832x480 without cropping, with the interpolation cosmos-framework's
``direct_resize`` uses for that kind of control, scaled to ``[-1, 1]`` and
encoded by the Wan 2.2 streaming encoder: one pixel frame for the first latent,
then four per latent. How a game colors its instance ids belongs to the game,
not here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import CONTROL_HINTS
from .prompt import HEIGHT, WIDTH

NEAREST_HINTS = frozenset({"seg", "edge"})
"""Controls resized by nearest neighbor: label palettes and thin edges stay intact."""


def resize_control(frames: Tensor, hint: str) -> Tensor:
    """Resize ``[T, 3, H, W]`` uint8 control frames to ``[T, 3, 480, 832]`` float in ``[0, 255]``.

    Matches cosmos-framework's ``direct_resize``: nearest neighbor without
    antialiasing for ``seg`` and ``edge``, bilinear with antialiasing for
    ``depth`` (kept fractional), bicubic with antialiasing for ``blur``
    (rounded back to whole levels). Frames already at 832x480 pass through.

    Raises:
        ValueError: ``hint`` is unknown, or the frames are not ``[T, 3, H, W]`` uint8.
    """
    if hint not in CONTROL_HINTS:
        raise ValueError(f"unknown control {hint!r}; use one of {CONTROL_HINTS}")
    if frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.uint8:
        raise ValueError(
            f"expected [T, 3, H, W] uint8 frames, got {tuple(frames.shape)} {frames.dtype}"
        )
    pixels = frames.float()
    if tuple(frames.shape[-2:]) == (HEIGHT, WIDTH):
        return pixels
    size = (HEIGHT, WIDTH)
    if hint in NEAREST_HINTS:
        return F.interpolate(pixels, size=size, mode="nearest")
    if hint == "depth":
        return F.interpolate(
            pixels, size=size, mode="bilinear", align_corners=False, antialias=True
        )
    resized = F.interpolate(
        pixels, size=size, mode="bicubic", align_corners=False, antialias=True
    )
    return resized.clamp(0, 255).round()


def to_model_range(pixels: Tensor) -> Tensor:
    """``[0, 255]`` pixels to the encoder's ``[-1, 1]``, with the same values on every device.

    The divisor is a tensor, not a Python number: CUDA turns division by a
    number into multiplication by its reciprocal, which rounds 111 of the 256
    levels differently from the CPU in float32 (one of them still differs in
    bfloat16), enough to change the control latents and so the generated clip.
    """
    return pixels / pixels.new_tensor(127.5) - 1.0


def read_control_video(path: str | Path, *, max_frames: int | None = None) -> Tensor:
    """Decode a control video file to ``[T, 3, H, W]`` uint8 RGB frames.

    Args:
        path: Video file.
        max_frames: Stop after this many frames.
    """
    from flashdreams.infra.runner_io import read_video_window_rgb

    video = read_video_window_rgb(path, frame_count=max_frames)
    return torch.from_numpy(video.copy()).permute(0, 3, 1, 2).contiguous()


def read_control_fps(path: str | Path) -> float:
    """Frame rate of a video file, from its container metadata."""
    import mediapy

    with mediapy.VideoReader(str(path)) as reader:
        return float(reader.metadata.fps)


def latent_frame_spans(num_frames: int, temporal_compression: int = 4) -> list[slice]:
    """Pixel-frame slices per latent: one frame, then ``temporal_compression`` each.

    A trailing partial group is dropped, so ``1 + 4n`` frames give ``1 + n``
    latents; :func:`padded_frame_indices` completes a clip to whole latents.
    """
    if num_frames < 1:
        raise ValueError(f"need at least one frame, got {num_frames}")
    spans = [slice(0, 1)]
    start = 1
    while start + temporal_compression <= num_frames:
        spans.append(slice(start, start + temporal_compression))
        start += temporal_compression
    return spans


def latent_count(num_frames: int, temporal_compression: int = 4) -> int:
    """Latents of a ``num_frames`` clip, the last one completed by padding: ``1 + ceil((T - 1) / 4)``."""
    if num_frames < 1:
        raise ValueError(f"need at least one frame, got {num_frames}")
    return 1 + -(-(num_frames - 1) // temporal_compression)


def padded_frame_indices(num_frames: int, target: int | None = None) -> list[int]:
    """Frame indices of a clip padded in time to ``target`` frames.

    cosmos-framework pads a Transfer request's control this way before encoding
    it (``pad_temporal_frames``: append the clip reversed, its last frame first,
    again while it is too short; repeat the frame of a one-frame clip), then
    trims the decoded output back to the clip's length.

    Args:
        num_frames: Frames of the clip.
        target: Frames after padding; by default whole latents,
            ``1 + 4 * (latent_count(num_frames) - 1)``.
    """
    if num_frames < 1:
        raise ValueError(f"need at least one frame, got {num_frames}")
    if target is None:
        target = 1 + 4 * (latent_count(num_frames) - 1)
    indices = list(range(num_frames))
    while len(indices) < target:
        pad = min(len(indices) - 1, target - len(indices))
        if pad <= 0:
            indices += [indices[-1]] * (target - len(indices))
            break
        indices += indices[::-1][:pad]
    return indices


class ControlEncoder:
    """Wan 2.2 streaming encoder for control frames, one latent per call.

    The first call takes one pixel frame, every later call
    ``temporal_compression`` frames; the encoder's causal cache carries the
    context between calls. :meth:`reset` starts a new clip.

    Args:
        checkpoint_path: Wan 2.2 VAE weights; the FlashDreams default when ``None``.
        device: Where to run the encoder.
        dtype: Encoder dtype.
        use_cuda_graph: Replay the four-frame calls as a CUDA graph (captured
            again in every clip, after two eager calls; CUDA only).
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_cuda_graph: bool = False,
    ) -> None:
        from cosmos_interactive._vae import wan22_vae_encoder_config

        if use_cuda_graph and torch.device(device).type != "cuda":
            raise ValueError(f"CUDA graphs need a CUDA device, not {device}")
        config = wan22_vae_encoder_config(
            dtype=dtype, checkpoint_path=checkpoint_path, use_cuda_graph=use_cuda_graph
        )
        self.encoder = config.setup().to(device)
        self.device = torch.device(device)
        self.dtype = dtype
        self.temporal_compression = int(self.encoder.temporal_compression_ratio)
        self.reset()

    def reset(self) -> None:
        """Forget the previous clip."""
        self._cache = self.encoder.initialize_autoregressive_cache()
        self._latents = 0

    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """Encode the pixel frames of the next latent to a ``[C, 1, h, w]`` latent.

        Args:
            frames: ``[T, 3, 480, 832]`` in ``[-1, 1]``: one frame for the first
                latent of a clip, ``temporal_compression`` frames after it.

        Raises:
            ValueError: The frame count does not fit this position in the clip.
        """
        expected = 1 if self._latents == 0 else self.temporal_compression
        if frames.ndim != 4 or frames.shape[0] != expected:
            raise ValueError(
                f"latent {self._latents} takes {expected} frames, got {tuple(frames.shape)}"
            )
        latent = self.encoder(
            frames.to(device=self.device, dtype=self.dtype)[None], cache=self._cache
        )
        self._latents += 1
        return latent[0].transpose(0, 1).contiguous()
