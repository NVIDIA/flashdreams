# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render prepared HD-map geometry on the driven or recorded camera path.

Controls stay on the device and are resized for the control encoder. Geometry,
palette and calibration come from the prepared bundle; no source-data loader
or reference renderer is used during a rollout.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from cosmos3_nano_sim_auto.impl.conditioning import camera_rig
from cosmos3_nano_sim_auto.impl.conditioning.driving import PoseSource
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    camera as camera_math,
)
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    prepared_scene,
)
from cosmos3_nano_sim_auto.inputs import VIEW_SIZES

__all__ = [
    "DrawnCamera",
    "LudusControls",
    "open_clip",
]

FED = VIEW_SIZES["480"]
"""``(width, height)`` every view is fed at, whatever size it was drawn at."""


@dataclass(frozen=True)
class DrawnCamera:
    """One camera of the rig, with the size it is drawn at and its poses.

    A camera is built for a resolution -- ``ftheta_camera`` scales the optical
    centre and the polynomial onto it -- so the size travels with the camera
    rather than with the render call, and the width table travels with it too:
    ``thinner`` is how far the drawing is from the camera's own pixels, and the
    widths scale by the same ratio or the lines come out the wrong weight.
    """

    name: str
    camera: Any
    """A ludus ``FThetaCamera``, built for :attr:`drawn`."""

    drawn: tuple[int, int]
    """``(width, height)`` the renderer is asked for."""

    thinner: float
    """:attr:`drawn`'s width over the camera's native width."""

    track: np.ndarray
    """``[T, 4, 4]`` camera-to-world, one a source frame."""


class LudusControls:
    """A clip's map, drawn for the cameras asked for, a frame span at a time.

    Holds the scene and the cameras; :meth:`render` is the
    :data:`~cosmos3_nano_sim_auto.impl.conditioning.control_provider.RigRenderFn`. The context is passed
    in rather than built here so that the ordering this is responsible for --
    which camera, which frame, at which timestamp and pose -- can be tested
    without a card. :func:`open_clip` is what builds the real one.
    """

    def __init__(
        self,
        context: Any,
        scene_id: int,
        cameras: Sequence[DrawnCamera],
        *,
        start_us: int,
        fps: float,
        widths: dict[int, float],
        device: torch.device | str = "cuda",
        frames: Sequence[int] | np.ndarray | None = None,
        note: Callable[[str], None] = lambda _: None,
        drive: PoseSource | None = None,
    ) -> None:
        """
        Args:
            context: a ludus ``LudusCudaTimestampedContext``, with the palette
                uploaded and the scene already on it.
            scene_id: what ``upload_scene`` returned for that scene.
            cameras: the rig, in the order the views are fed in.
            start_us: the clip's first timestamp, which frame 0 sits at.
            fps: the rate the recorded track was sampled at.
            widths: the primitive width table at the cameras' native size.
            device: where the query tensors are built and the frames come back.
            note: told what each span cost, since a drawn span happens inside
                somebody else's stage -- the prefill's, or a chunk's generate --
                and is otherwise invisible in both.
            drive: where the poses come from instead of the recorded track, be
                that a driver reading the keys or a host application handing
                over a path it has already worked out. The
                cameras keep the places on the car they were recorded in, taken
                from where the run begins, and the whole rig follows the one
                pose this gives. Timestamps still come from the clip whichever
                source is used: the driver moves the car, not the afternoon.
            frames: which of the clip's frames the run's frames stand at, or
                ``None`` for the clip's own. A run mirroring a recorded-scene clip is cut
                to a different window at a different rate, so its frame 0 is
                somewhere inside the track and its frames may sample the track
                twice over or skip through it.

        Raises:
            ValueError: no cameras, their tracks are different lengths, or the
                mapping names frames the track does not have.
        """
        if not cameras:
            raise ValueError("a rig needs at least one camera.")
        lengths = {len(one.track) for one in cameras}
        if len(lengths) != 1:
            raise ValueError(
                "the rig's cameras have tracks of different lengths: "
                + ", ".join(f"{one.name} {len(one.track)}" for one in cameras)
            )
        recorded = len(cameras[0].track)
        self._map = (
            np.arange(recorded)
            if frames is None
            else np.asarray(frames, dtype=np.int64)
        )
        outside = self._map[(self._map < 0) | (self._map >= recorded)]
        if outside.size:
            raise ValueError(
                f"{outside.size} of {self._map.size} frames fall outside the "
                f"track's {recorded} poses, first at {int(outside[0])}. The run is "
                "cut to a window this clip does not cover."
            )
        self._context = context
        self._scene_id = scene_id
        self._cameras = list(cameras)
        self._start_us = start_us
        self._fps = float(fps)
        self._widths = dict(widths)
        self._device = torch.device(device)
        self._note = note
        self._drive = drive
        # Where each camera sits relative to the one the driver steers, read off
        # the frame the run starts at. The rig is bolted together, so this holds
        # for every frame; taking it from the start rather than averaging keeps
        # a driven run's first frame identical to the recorded one's.
        self._bolted = [
            np.linalg.inv(cameras[0].track[self._map[0]]) @ one.track[self._map[0]]
            for one in cameras
        ]

    @property
    def cameras(self) -> tuple[str, ...]:
        """The rig's camera names, in view order."""
        return tuple(one.name for one in self._cameras)

    @property
    def frames(self) -> int:
        """Run frames this can draw, which is what the mapping covers."""
        return int(self._map.size)

    @property
    def fps(self) -> float:
        """The rate the track was sampled at, which frames are timed by."""
        return self._fps

    def render(
        self, first: int, last: int, views: Sequence[int] | None = None
    ) -> Tensor:
        """Draw pixel frames ``[first, last)`` for every camera, or for ``views``.

        Returns:
            ``[V*T, H, W, 3]`` uint8 on :attr:`device`, view-outer and
            frame-inner at the size the model is fed, which is what
            :data:`~cosmos3_nano_sim_auto.impl.conditioning.control_provider.RigRenderFn` asks for.

        Past the end of the recorded track a driven run holds the scene clock
        at its last frame: other traffic and the light states stop where the
        recording stopped, while the road, the buildings and the lane markings
        carry on being drawn from wherever the driver has steered to. That is
        what lets a driven run outlive its clip. A recorded run has no driver
        to move the ego, so the same thing would be the last pose drawn forever
        and it is refused instead.

        ``views`` draws only those cameras, in that order, so a rank that encodes
        one camera need not draw the rest. The driver still advances once for
        the span on every rank, which keeps every rank's car in the same place.

        Raises:
            ValueError: the span is empty, or runs past the recorded track on a
                run with nobody driving.
        """
        if first < 0 or first >= last:
            raise ValueError(f"[{first}, {last}) is not a frame span.")
        if last > self.frames and self._drive is None:
            raise ValueError(
                f"[{first}, {last}) runs past the {self.frames} frames this "
                "clip's track covers. Ask for fewer frames, or drive it."
            )
        spanned = self._map[
            np.minimum(np.arange(first, last), self.frames - 1)
            if last > self.frames
            else slice(first, last)
        ]
        started = time.perf_counter()
        # The driver is asked once for the span, not once a camera: its poses
        # are the sum of the driving so far and it will not give them twice.
        driven = None if self._drive is None else self._drive.poses(first, last)
        drawn = range(len(self._cameras)) if views is None else views
        if not drawn:
            # A rank that owns no camera still had to move the car above.
            return torch.empty((0, 0, 0, 3), dtype=torch.uint8, device=self._device)
        frames = torch.cat(
            [
                self._drawn(self._cameras[view], spanned, driven, self._bolted[view])
                for view in drawn
            ]
        )
        # Synchronised before the clock is read, because a render call returns
        # as soon as it is queued. Unsynchronised, this would report the launch
        # and read as though drawing were free, which is the wrong thing to
        # believe about the one stage of a chunk that is ours.
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._note(
            f"  drew frames [{first}, {last}) for {len(drawn)} camera(s) "
            f"in {time.perf_counter() - started:.2f}s"
        )
        return frames

    def _drawn(
        self,
        camera: DrawnCamera,
        spanned: np.ndarray,
        driven: np.ndarray | None,
        bolted: np.ndarray,
    ) -> Tensor:
        """One camera's span, drawn at its own size and shrunk to the model's.

        A call draws at one size and through one width table, so a camera is
        uploaded with its own before it is asked for. One call a camera is more
        calls than the rig needs -- cameras sharing a lens shape could go in
        one -- and correctness does not depend on the grouping, only the cost.
        """
        self._context.upload_cameras([camera.camera])
        self._context.set_line_widths(
            wireframe=camera_math.OUTLINE_WIDTH * camera.thinner
        )
        self._context.upload_width_table(
            {prim: width * camera.thinner for prim, width in self._widths.items()}
        )

        count = int(spanned.size)
        images = self._context.render(
            scene_ids=torch.full(
                (count,), self._scene_id, dtype=torch.int32, device=self._device
            ),
            camera_ids=torch.zeros(count, dtype=torch.int32, device=self._device),
            timestamps_us=torch.tensor(
                [self._at(int(frame)) for frame in spanned],
                dtype=torch.int64,
                device=self._device,
            ),
            camera_type_ids=torch.zeros(count, dtype=torch.int32, device=self._device),
            camera_poses=self._poses(camera, spanned, driven, bolted),
            resolution=(camera.drawn[1], camera.drawn[0]),
        )
        return _shrunk(images, FED)

    def _at(self, frame: int) -> int:
        """The timestamp of one of the clip's own frames, in microseconds."""
        return self._start_us + round(frame * 1e6 / self._fps)

    def _poses(
        self,
        camera: DrawnCamera,
        spanned: np.ndarray,
        driven: np.ndarray | None,
        bolted: np.ndarray,
    ) -> Tensor:
        """``[T, 4, 4]`` world-to-camera for those frames, in one transfer.

        The whole span through ``camera_pose`` at once, which takes a stack,
        rather than a frame at a time: the inverse is the cost here and NumPy
        batches it.

        Driven, this camera is where the driven one is plus where it sits on the
        car. Recorded, it is what the rig wrote down, which for the run's first
        frame is the same pose either way.
        """
        if driven is None:
            poses = camera_math.camera_pose(camera.track[spanned])
        else:
            poses = camera_math.camera_pose(driven @ bolted)
        return torch.from_numpy(np.ascontiguousarray(poses)).to(self._device)


def _shrunk(images: Tensor, fed: tuple[int, int]) -> Tensor:
    """``[T, H, W, 4]`` uint8 as drawn -> ``[T, fed_h, fed_w, 3]`` uint8.

    Bilinear and without antialiasing, which is what OpenCV's default does and
    therefore what the file path fed the model when this drawing was judged
    against it. A downscale of better than two to one would rather have an area
    filter, and the difference on hairline map markings is visible, so changing
    it is a decision about what the model is fed and not a tidy-up.

    Alpha goes before the resize rather than after: the renders are opaque, the
    control encoder takes three channels, and a fourth costs a third of the
    filtering for nothing.
    """
    width, height = fed
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"expected [T, H, W, C>=3] frames, got {tuple(images.shape)}.")
    if images.shape[1:3] == (height, width):
        return images[..., :3].contiguous()
    planes = images[..., :3].permute(0, 3, 1, 2).to(torch.float32)
    smaller = F.interpolate(
        planes, size=(height, width), mode="bilinear", align_corners=False
    )
    return (
        smaller.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    )


def open_clip(
    clip: Path,
    cameras: Sequence[str],
    *,
    device: torch.device | str = "cuda",
    against: Path | None = None,
    start_frame: int = 0,
    wanted: int | None = None,
    drive: PoseSource | None = None,
    note: Callable[[str], None] = lambda _: None,
    span_note: Callable[[str], None] | None = None,
    prepared: prepared_scene.PreparedScene | None = None,
    requested_fps: float | None = None,
) -> LudusControls:
    """Upload a prepared scene and render its recorded or supplied rig path."""
    if not cameras:
        raise ValueError("no cameras to draw")
    camera_rig.view_ids(cameras)
    if not (clip / "scene.json").is_file():
        raise ValueError(
            "Expected a prepared scene.json bundle; prepare raw scenes with multiview-scene-tools"
        )
    prepared = prepared or prepared_scene.load(clip)
    if prepared.root != clip.resolve():
        raise ValueError("Prepared scene belongs to a different directory")
    if against is not None and against.resolve() != prepared.root:
        raise ValueError("Prepared scene must use its own matching opening sample")
    if any(name not in prepared.tracks for name in cameras):
        raise ValueError("Requested camera is absent from the prepared scene")
    mapping = prepared.frame_mapping(start_frame, wanted)
    start_us, end_us = (
        int(prepared.manifest["start_us"]),
        int(prepared.manifest["end_us"]),
    )
    fps = camera_math.frame_rate(prepared.manifest["frames"], start_us, end_us)
    if requested_fps is not None and not np.isclose(
        requested_fps, fps, rtol=0, atol=1e-3
    ):
        raise ValueError(f"Prepared scene requires {fps:.6f} FPS")
    from ludus_renderer import LudusCudaTimestampedContext

    context = LudusCudaTimestampedContext(device=device)
    prepared_scene.configure(context)
    context.upload_color_palette(prepared.palette)
    scene_id = context.upload_scene(prepared.upload_data(device))
    built = []
    for name in cameras:
        intrinsics = prepared.intrinsics(name)
        native = (int(intrinsics[2]), int(intrinsics[3]))
        drawn = camera_math.smoothable(native)
        built.append(
            DrawnCamera(
                name=name,
                camera=camera_math.ftheta_camera(intrinsics, device, resolution=drawn),
                drawn=drawn,
                thinner=drawn[0] / native[0],
                track=prepared.tracks[name],
            )
        )
    note(f"clip {prepared.manifest['source_clip_id']}")
    note(f"  {prepared.manifest['frames']} frames of it, so {fps:.1f}fps")
    return LudusControls(
        context,
        scene_id,
        built,
        start_us=start_us,
        fps=fps,
        widths=prepared.widths,
        device=device,
        frames=mapping,
        note=note if span_note is None else span_note,
        drive=drive,
    )
