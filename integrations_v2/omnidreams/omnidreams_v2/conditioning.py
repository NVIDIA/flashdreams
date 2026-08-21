# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a run's HDMap conditioning comes from."""

from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor

from flashdreams.infra.runner_io import load_first_frame_tensor, load_video_tensor
from flashdreams.runtime_v2.user_input_events import UserInputEvents

_PIXEL_DTYPE = torch.bfloat16
"""What the model reads its conditioning pixels as."""


class HDMapSource(Protocol):
    """One run's worth of HDMap conditioning, a chunk at a time.

    This model generates from rendered road layout rather than from a prompt
    alone, so every step needs a chunk of HDMap pixels to condition on. A
    recording has them already; a renderer draws them from wherever the driver
    steered. Both answer the same questions, so a session does not know which
    of them it was given.

    Not a client window and not an :class:`InputSource`: this produces model
    input at the rate the model generates, and has to know how long the chunk
    it is being asked for is. A window supplies what a person did, which is a
    different thing arriving at a different rate. A source that renders is
    given those events and turns them into geometry.
    """

    @property
    def view_names(self) -> tuple[str, ...]:
        """Cameras this supplies, in the order its chunks stack them."""
        ...

    def open(self) -> None:
        """Load or start whatever produces the frames.

        Called once, from ``ISession.init``, so the cost of reading a video or
        starting a renderer lands with the rest of a run's startup.
        """
        ...

    def first_frame(self) -> Tensor:
        """Return the frame the run continues from.

        Returns:
            Pixels as ``[B, V, 1, 3, H, W]`` in ``[-1, 1]``.
        """
        ...

    def has_frames(self, frame_count: int) -> bool:
        """Report whether a chunk of ``frame_count`` more frames can be produced.

        Asked before each step, so a run ends on the boundary rather than on a
        half-conditioned chunk. A renderer with no end to it always says yes.
        """
        ...

    def next_chunk(self, frame_count: int, events: UserInputEvents) -> Tensor:
        """Return the next chunk, advancing this source past it.

        Args:
            frame_count: Frames the model is about to generate, which is how
                many frames of conditioning it needs.
            events: Input since the previous step. Ignored by a source that is
                replaying something already recorded.

        Returns:
            Pixels as ``[B, V, T, 3, H, W]`` in ``[-1, 1]``, with ``T`` equal to
            ``frame_count``.
        """
        ...

    def reset(self) -> None:
        """Start this source over, so the run can generate from the top."""
        ...

    def close(self) -> None:
        """Release whatever this holds."""
        ...


class SceneRenderer(Protocol):
    """Frames of road layout, drawn on demand along a scene's timeline.

    The part of a rendered run that needs a GPU and a scene on disk, kept
    behind a seam so the bookkeeping around it -- how far through the drive a
    run is, and what the model wants its pixels to look like -- can be tested
    without either.
    """

    @property
    def frame_count(self) -> int:
        """Frames this can draw, which is how long a run against it lasts."""
        ...

    def open(self) -> None:
        """Load the scene and get the rasterizer ready to draw it."""
        ...

    def render(self, start: int, count: int) -> Tensor:
        """Draw ``count`` frames of the timeline, beginning at ``start``.

        Returns:
            Pixels as ``[T, H, W, 3]``, RGB bytes.
        """
        ...

    def close(self) -> None:
        """Release the scene and the rasterizer."""
        ...


class PrecomputedHDMapSource:
    """HDMap conditioning read from recorded video, one file per camera.

    What a reproducible run uses: the same clip conditions every run, so two
    runs of one command are comparable. The recording is also what says how
    long the run is, since generating past the end of it would have nothing to
    condition on.
    """

    def __init__(
        self,
        *,
        hdmap_video_paths: tuple[Path, ...],
        first_frame_paths: tuple[Path, ...],
        view_names: tuple[str, ...],
        pixel_width: int,
        pixel_height: int,
        device: str,
    ) -> None:
        """
        Args:
            hdmap_video_paths: HDMap video per camera, in camera order.
            first_frame_paths: Frame to continue from per camera, in the same
                order. An image or a video, of which frame zero is taken.
            view_names: Camera labels, in the same order.
            pixel_width: Width to resize the conditioning to, which is the
                width the run generates at.
            pixel_height: Height to resize it to.
            device: Device to load onto, alongside the model.
        """
        self._hdmap_video_paths = hdmap_video_paths
        self._first_frame_paths = first_frame_paths
        self._view_names = view_names
        self._pixel_width = pixel_width
        self._pixel_height = pixel_height
        self._device = torch.device(device)
        self._hdmap: Tensor | None = None
        self._cursor = 0

    @property
    def view_names(self) -> tuple[str, ...]:
        return self._view_names

    def open(self) -> None:
        """Decode every camera's HDMap video into one tensor.

        The whole clip is held, rather than decoded a chunk at a time, because
        a reset has to replay it and decoding is the slow part.
        """
        videos = [self._load_video(path) for path in self._hdmap_video_paths]
        # [T, C, H, W] per camera, stacked into [B=1, V, T, C, H, W].
        self._hdmap = torch.stack(videos, dim=0).unsqueeze(0)

    def first_frame(self) -> Tensor:
        frames = [
            load_first_frame_tensor(
                path,
                pixel_height=self._pixel_height,
                pixel_width=self._pixel_width,
                device=self._device,
                dtype=_PIXEL_DTYPE,
                allow_video=True,
            )
            for path in self._first_frame_paths
        ]
        return torch.stack(frames, dim=0).unsqueeze(0)

    def has_frames(self, frame_count: int) -> bool:
        return self._cursor + frame_count <= self._frames().shape[2]

    def next_chunk(self, frame_count: int, events: UserInputEvents) -> Tensor:
        """Return the next ``frame_count`` frames of the recording.

        Args:
            frame_count: Frames to return.
            events: Ignored. A recording plays the drive it recorded, whatever
                anyone does while watching it.

        Raises:
            RuntimeError: The recording has fewer frames left than that, which
                :meth:`has_frames` is asked in order to avoid.
        """
        del events
        hdmap = self._frames()
        end = self._cursor + frame_count
        if end > hdmap.shape[2]:
            raise RuntimeError(
                f"Asked for frames [{self._cursor}, {end}) of an HDMap "
                f"recording {hdmap.shape[2]} frames long."
            )
        chunk = hdmap[:, :, self._cursor : end]
        self._cursor = end
        return chunk

    def reset(self) -> None:
        """Rewind to the start of the recording."""
        self._cursor = 0

    def close(self) -> None:
        self._hdmap = None

    def _frames(self) -> Tensor:
        """Return the loaded recording.

        Raises:
            RuntimeError: This was never opened, or has been closed.
        """
        if self._hdmap is None:
            raise RuntimeError(
                f"{type(self).__name__}.open() must run before the recording "
                "is read, and it cannot be read after close()."
            )
        return self._hdmap

    def _load_video(self, path: Path) -> Tensor:
        """Load and resize one camera's HDMap video to ``[T, C, H, W]``."""
        return load_video_tensor(
            path,
            pixel_height=self._pixel_height,
            pixel_width=self._pixel_width,
            device=self._device,
            dtype=_PIXEL_DTYPE,
        )


class RenderedHDMapSource:
    """HDMap conditioning drawn a chunk at a time by a renderer.

    The reason for drawing rather than replaying is that a drive can then go
    somewhere the recording never went. Nothing steers yet, so what this
    produces is the drive the scene recorded -- the same road as the matching
    recording, which is what makes the two comparable while this is new.

    One camera, because the renderer draws one. A multi-camera run would stack
    what several renderers drew, which is where this and the renderer seam grow
    together rather than here alone.
    """

    def __init__(
        self,
        *,
        renderer: SceneRenderer,
        first_frame_path: Path,
        view_name: str,
        pixel_width: int,
        pixel_height: int,
        device: str,
    ) -> None:
        """
        Args:
            renderer: Draws the road layout. Owned by this source, which opens
                and closes it with the run.
            first_frame_path: Frame the run continues from, which for a scene is
                the recorded capture the drawn layout starts at.
            view_name: Camera label, as the model spells it.
            pixel_width: Width the run generates at, which the layout is drawn
                at so the two line up.
            pixel_height: Height the run generates at.
            device: Device to hand the model its pixels on.
        """
        self._renderer = renderer
        self._first_frame_path = first_frame_path
        self._view_name = view_name
        self._pixel_width = pixel_width
        self._pixel_height = pixel_height
        self._device = torch.device(device)
        self._cursor = 0

    @property
    def view_names(self) -> tuple[str, ...]:
        return (self._view_name,)

    def open(self) -> None:
        """Load the scene, which is where the run's startup cost mostly is."""
        self._renderer.open()

    def first_frame(self) -> Tensor:
        frame = load_first_frame_tensor(
            self._first_frame_path,
            pixel_height=self._pixel_height,
            pixel_width=self._pixel_width,
            device=self._device,
            dtype=_PIXEL_DTYPE,
            allow_video=True,
        )
        return frame.unsqueeze(0).unsqueeze(0)

    def has_frames(self, frame_count: int) -> bool:
        return self._cursor + frame_count <= self._renderer.frame_count

    def next_chunk(self, frame_count: int, events: UserInputEvents) -> Tensor:
        """Draw the next ``frame_count`` frames of road layout.

        Args:
            frame_count: Frames to draw.
            events: Ignored. This is where steering would enter, turning what a
                driver did into where the next chunk is drawn from; until then a
                rendered run follows the drive the scene recorded.

        Raises:
            RuntimeError: The scene has fewer frames left than that, which
                :meth:`has_frames` is asked in order to avoid.
        """
        del events
        available = self._renderer.frame_count
        end = self._cursor + frame_count
        if end > available:
            raise RuntimeError(
                f"Asked for frames [{self._cursor}, {end}) of a scene "
                f"{available} frames long."
            )
        frames = self._renderer.render(self._cursor, frame_count)
        self._cursor = end
        return _to_model_pixels(frames, device=self._device).unsqueeze(0).unsqueeze(0)

    def reset(self) -> None:
        """Return to the start of the drive."""
        self._cursor = 0

    def close(self) -> None:
        self._renderer.close()


def _to_model_pixels(frames: Tensor, *, device: torch.device) -> Tensor:
    """Convert drawn ``[T, H, W, 3]`` RGB bytes to model pixels ``[T, 3, H, W]``.

    A rasterizer hands back bytes in the layout an image viewer wants. The model
    reads signed pixels in the layout a convolution wants, which is what the
    recorded path's video loader already returns, so the conversion the two
    paths share happens here.
    """
    pixels = frames.to(device=device, dtype=_PIXEL_DTYPE)
    return pixels.permute(0, 3, 1, 2) / 127.5 - 1.0
