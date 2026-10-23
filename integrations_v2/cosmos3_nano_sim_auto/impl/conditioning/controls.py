# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load local HD-map control videos or draw synthetic diagnostic controls.

Both paths return ``[T, 3, H, W]`` tensors in ``[-1, 1]``. Synthetic controls
exercise inference plumbing; use a prepared scene for interactive driving.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from cosmos3_nano_sim_auto.inputs import MultiviewSample

_ROAD = (0.35, 0.35, 0.40)
"""Road surface, mid grey. Control renders are flat-shaded, so these are too."""

_LANE = (0.95, 0.95, 0.80)
"""Lane markings, off-white."""

_BOX = (0.90, 0.30, 0.25)
"""The moving obstacle, red. Bounding boxes are what the real controls draw."""

_SKY = (0.08, 0.09, 0.12)
"""Everything that is not road."""


Rectangle = tuple[int, int]
"""A half-open ``(start, stop)`` span on one axis."""


def _fill(
    clip: Tensor,
    frame: int,
    rows: Rectangle,
    columns: Rectangle,
    colour: tuple[float, float, float],
) -> None:
    """Paint one axis-aligned rectangle, clipped to the canvas.

    Clipping rather than validating, because the wedge and the dashes are drawn
    from a centre and a half-width that both run off the edge by design.
    """
    _frames, _channels, height, width = clip.shape
    top, bottom = max(0, rows[0]), min(height, rows[1])
    left, right = max(0, columns[0]), min(width, columns[1])
    if top >= bottom or left >= right:
        return
    for channel, value in enumerate(colour):
        clip[frame, channel, top:bottom, left:right] = value * 2.0 - 1.0


def synthetic_control_clip(
    *,
    pixel_frames: int,
    pixel_height: int,
    pixel_width: int,
    view: int = 0,
    num_views: int = 1,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> Tensor:
    """A stand-in control clip, ``[T, 3, H, W]`` in ``[-1, 1]``.

    Args:
        pixel_frames: Frames to draw.
        pixel_height, pixel_width: One view's size.
        view: Which camera this is. Sets the road's heading and the obstacle's
            phase so the tracks differ from one another; identical controls
            across views would hide a wrong view position or view id.
        num_views: cameras in the rig, which ``view`` is spread across.
        dtype, device: where to build it.

    Raises:
        ValueError: a dimension is not positive, or ``view`` is outside the rig.
    """
    if pixel_frames < 1 or pixel_height < 1 or pixel_width < 1:
        raise ValueError(
            f"clip dimensions must be positive, got {pixel_frames}x{pixel_width}x{pixel_height}."
        )
    if num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}.")
    if not 0 <= view < num_views:
        raise ValueError(f"view must lie in [0, {num_views}), got {view}.")

    clip = torch.empty(
        pixel_frames, 3, pixel_height, pixel_width, dtype=dtype, device=device
    )
    for channel, value in enumerate(_SKY):
        clip[:, channel] = value * 2.0 - 1.0

    heading = 2.0 * math.pi * view / num_views
    vanishing = int(pixel_width * (0.5 + 0.28 * math.sin(heading)))
    horizon = int(pixel_height * 0.45)
    half_base = int(pixel_width * 0.42)
    dash_height = max(2, pixel_height // 24)
    dash_period = max(2 * dash_height, pixel_height // 8)
    box_height = max(4, pixel_height // 10)
    box_width = max(4, pixel_width // 12)

    for frame in range(pixel_frames):
        # The road: a wedge from the vanishing point to the bottom edge, drawn a
        # row at a time so its width interpolates without needing a rasterizer.
        for row in range(horizon, pixel_height):
            depth = (row - horizon) / max(1, pixel_height - horizon)
            half = int(half_base * depth)
            centre = int(vanishing + (pixel_width / 2 - vanishing) * depth)
            _fill(clip, frame, (row, row + 1), (centre - half, centre + half), _ROAD)

        # Lane dashes scroll toward the camera. The only motion in here, and so
        # the only thing a break between chunks would show up in.
        scroll = (frame * dash_height) % dash_period
        for row in range(horizon + scroll, pixel_height, dash_period):
            depth = (row - horizon) / max(1, pixel_height - horizon)
            centre = int(vanishing + (pixel_width / 2 - vanishing) * depth)
            half = max(1, int(pixel_width * 0.012 * (0.3 + depth)))
            rows = (row, row + dash_height)
            _fill(clip, frame, rows, (centre - half, centre + half), _LANE)

        # One obstacle crossing left to right, out of phase per camera.
        sweep = (frame / max(1, pixel_frames - 1) + view / num_views) % 1.0
        left = int(sweep * (pixel_width - box_width))
        top = int(pixel_height * 0.62)
        _fill(clip, frame, (top, top + box_height), (left, left + box_width), _BOX)

    return clip


def generated_control_clips(
    sample: MultiviewSample,
    *,
    pixel_frames: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> list[Tensor]:
    """One stand-in clip per camera of ``sample``, in rig order."""
    width, height = sample.view_size
    return [
        synthetic_control_clip(
            pixel_frames=pixel_frames,
            pixel_height=height,
            pixel_width=width,
            view=view,
            num_views=sample.num_views,
            dtype=dtype,
            device=device,
        )
        for view in range(sample.num_views)
    ]


def loaded_control_clips(
    paths: Sequence[str | Path],
    *,
    pixel_frames: int,
    pixel_height: int,
    pixel_width: int,
    start_frame: int = 0,
    loop: bool = False,
    loop_frames: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> list[Tensor]:
    """Read one video track per camera, resized to the view size and cut to length.

    Control renders and real footage are read identically, so an image2video
    clip's opening frames come through here too. A clip too short for the
    window is an error unless ``loop`` asks for it to repeat.

    Args:
        paths: One existing local video per camera, in rig order.
        pixel_frames: Frames to keep from each clip.
        pixel_height, pixel_width: One view's size.
        start_frame: Source frame to cut from, the same for every camera. Not
            always 0, since the caption describes a particular span of the
            drive. See
            :meth:`~cosmos3_nano_sim_auto.inputs.MultiviewSample.offset_for`.
        loop: Repeat each track from ``start_frame`` until it covers
            ``pixel_frames``, the way the reference's long rollouts replay a
            clip's controls. The seam is a jump the encoder sees as one.
        loop_frames: The length every track repeats at under ``loop``, so the
            cameras stay in step past the seam. Defaults to the shortest of
            ``paths``; pass the whole rig's when ``paths`` is only some of it.
        dtype, device: Where to put the result.

    Raises:
        FileNotFoundError: a path is not on disk.
        ValueError: ``start_frame`` is negative, or a clip is too short for the
            window asked of it.
    """
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}.")
    from flashdreams.infra.runner_io import (
        IMAGE_SUFFIXES,
        load_first_frame_tensor,
        load_video_tensor,
    )

    require_local_files(paths)
    if loop and loop_frames is None:
        loop_frames = available_pixel_frames(paths, start_frame=start_frame)
    clips: list[Tensor] = []
    for path in paths:
        resolved = Path(path)
        if resolved.suffix.lower() in IMAGE_SUFFIXES:
            if start_frame != 0 or pixel_frames != 1:
                raise ValueError(
                    "An opening still image supplies exactly one frame at index 0"
                )
            clips.append(
                load_first_frame_tensor(
                    resolved,
                    pixel_height=pixel_height,
                    pixel_width=pixel_width,
                    device=torch.device(device),
                    dtype=dtype,
                ).contiguous()
            )
            continue
        clip = load_video_tensor(
            resolved,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            device=torch.device(device),
            dtype=dtype,
            start_frame=start_frame,
            frame_count=min(pixel_frames, loop_frames)
            if loop and loop_frames is not None
            else pixel_frames,
        )
        if loop and clip.shape[0] < pixel_frames:
            clip = clip.repeat(
                -(-pixel_frames // clip.shape[0]), *[1] * (clip.ndim - 1)
            )
            clip = clip[:pixel_frames]
        clips.append(clip.contiguous())
    return clips


def require_local_files(paths: Sequence[str | Path]) -> None:
    """Raise naming every path that is not on disk.

    Raises:
        FileNotFoundError: a path is not on disk.
    """
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} are not local files. "
            "Download control videos before loading them."
        )


def available_pixel_frames(paths: Sequence[str | Path], *, start_frame: int = 0) -> int:
    """Pixel frames every one of these tracks can supply from ``start_frame``.

    Read from container metadata rather than by decoding, since this is asked
    before a run starts to decide how long it will be, and decoding eleven
    clips to count them is most of a prefill.

    The answer is cut to the ``1 + 4n`` grid the encoder wants, so it is a
    length a run can be given rather than a frame count to round afterwards.

    Raises:
        FileNotFoundError: a path is not on disk.
        ValueError: ``start_frame`` is negative, or nothing is left after it.
    """
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}.")
    import mediapy as media

    require_local_files(paths)
    shortest = None
    for path in paths:
        with media.VideoReader(str(path)) as reader:
            counted = int(reader.num_images)
        shortest = counted if shortest is None else min(shortest, counted)
    if shortest is None:
        raise ValueError("no control tracks to measure.")
    left = shortest - start_frame
    if left < 1:
        raise ValueError(
            f"the shortest track is {shortest} frames, which has nothing at or "
            f"after frame {start_frame}."
        )
    return 1 + 4 * ((left - 1) // 4)


def write_control_clip(clip: Tensor, path: str | Path, *, fps: float = 30.0) -> Path:
    """Write a clip to mp4, for looking at what the model was conditioned on."""
    from flashdreams.infra.runner_io import write_video_tensor

    return write_video_tensor(clip, path, fps=fps, layout="tchw")


def writable(frames: Tensor) -> Tensor:
    """Bytes as the ``[-1, 1]`` :func:`write_control_clip` reads.

    Half a step up before the scaling, because the writer truncates where this
    rounded and every shade would otherwise come back one darker than it went
    in. Drawn pixels passed straight through would make a map dimmer than the
    one that was compared, which reads as a fidelity problem in the drawing.

    Raises:
        ValueError: not bytes, so the caller already has some range in mind and
            this would be the second scaling of the same pixels.
    """
    if frames.dtype != torch.uint8:
        raise ValueError(f"expected uint8 pixels, got {frames.dtype}.")
    return frames.to(torch.float32).add_(0.5).div_(127.5).sub_(1.0)
