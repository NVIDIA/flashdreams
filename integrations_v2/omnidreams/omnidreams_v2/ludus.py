# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Road layout drawn by the Ludus rasterizer, from a scene on disk."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class LudusSceneRenderer:
    """Draws one camera's view of a scene's road layout, a chunk at a time.

    A recording of the layout fixes the drive before the run starts. Drawing it
    does not, which is the point of this even while nothing steers: the geometry
    is produced beside the frames it conditions, so it can eventually follow
    wherever a driver goes. Until then the poses come from the drive the scene
    recorded, which makes a drawn run and the recording of the same scene two
    views of one drive.

    Everything Ludus is imported when the scene is loaded rather than when this
    module is, so a run that never draws anything -- and a test on a machine
    with no GPU -- does not pay for a rasterizer it will not use.
    """

    def __init__(
        self,
        *,
        scene_path: Path,
        camera: str,
        view_start_us: int,
        frames_per_second: int,
        pixel_width: int,
        pixel_height: int,
        device: str,
    ) -> None:
        """
        Args:
            scene_path: Scene archive to draw, holding the road layout and the
                drive recorded along it.
            camera: Camera to draw from, spelled the way the scene spells it.
            view_start_us: Timestamp to start drawing at, which is when the
                frame the run continues from was captured. Starting anywhere
                else would show the model a road it is not looking at.
            frames_per_second: Rate to draw at, matching the rate the model
                generates at. The recorded drive is resampled onto it.
            pixel_width: Width to draw at, which is the width the run
                generates at.
            pixel_height: Height to draw at.
            device: Device holding the scene and the drawn frames.
        """
        self._scene_path = scene_path
        self._camera = camera
        self._view_start_us = view_start_us
        self._frames_per_second = frames_per_second
        self._pixel_width = pixel_width
        self._pixel_height = pixel_height
        self._device = torch.device(device)
        self._context: Any = None
        self._scene_id: int | None = None
        self._camera_id: int | None = None
        self._camera_type: int | None = None
        self._ego_tracks: Any = None
        self._sensor_to_rig: Tensor | None = None
        self._timestamps: Tensor | None = None

    @property
    def frame_count(self) -> int:
        """Frames of recorded drive there are to draw.

        Zero until the scene is loaded, since the length of the drive is
        something the scene says.
        """
        return 0 if self._timestamps is None else int(self._timestamps.shape[0])

    def open(self) -> None:
        """Load the scene, upload it, and lay out the timeline to draw along.

        Raises:
            ValueError: The scene has no such camera, or no drive left after the
                frame the run continues from.
        """
        from ludus_renderer import LudusCudaTimestampedContext, load_scene
        from ludus_renderer.render_utils import SceneAdapter
        from ludus_renderer.torch.ops import CAMERA_TYPE_REGULAR

        scene = load_scene(
            self._scene_path,
            device=self._device,
            # Drawn at the size the run generates at, which scales the camera's
            # intrinsics with it. Drawing at a size the intrinsics disagree with
            # puts the road in the wrong place rather than failing.
            target_resolution=(self._pixel_width, self._pixel_height),
            # The ego trajectory is drawn as a path along the road and the ego
            # vehicle as a box in front of the camera. Neither is part of the
            # layout this model was conditioned on.
            include_ego_trajectory=False,
            include_ego_obstacle=False,
        )
        if self._camera not in scene.camera_name_to_id:
            available = ", ".join(sorted(scene.camera_name_to_id))
            raise ValueError(
                f"Scene {self._scene_path} has no camera {self._camera!r}. "
                f"It has: {available}."
            )

        self._timestamps = self._timeline(scene.ego_track.timestamps)
        self._ego_tracks = SceneAdapter(scene).ego_tracks
        self._sensor_to_rig = scene.sensor_to_rig[self._camera].to(self._device)
        self._camera_type = CAMERA_TYPE_REGULAR

        context = LudusCudaTimestampedContext(device=self._device)
        # Every camera, so a camera's own index is what identifies it, which is
        # what the scene's mapping already holds.
        context.upload_cameras(list(scene.cameras))
        self._camera_id = scene.camera_name_to_id[self._camera]
        self._scene_id = context.upload_scene(scene.timestamped_scene)
        self._context = context

    def render(self, start: int, count: int) -> Tensor:
        """Draw ``count`` frames of the drive, beginning at ``start``.

        Returns:
            Pixels as ``[T, H, W, 3]``, RGB bytes on this renderer's device.

        Raises:
            RuntimeError: :meth:`open` has not run yet.
        """
        context = self._context
        timestamps = self._timestamps
        if context is None or timestamps is None:
            raise RuntimeError(
                f"{type(self).__name__}.open() must run before render()."
            )
        chunk = timestamps[start : start + count]
        images = context.render_uniform(
            scene_id=self._scene_id,
            camera_id=self._camera_id,
            timestamps_us=chunk,
            camera_type_id=self._camera_type,
            camera_poses=self._camera_poses(chunk),
            resolution=(self._pixel_height, self._pixel_width),
        )
        # The rasterizer draws onto an opaque background, so the alpha it
        # reports is of no use to a model reading three channels.
        frames = images[:, :, :, :3]
        if context.needs_vflip:
            frames = frames.flip(1)
        return frames.contiguous()

    def close(self) -> None:
        """Drop the scene, and the device memory it was holding."""
        context = self._context
        self._context = None
        self._timestamps = None
        self._ego_tracks = None
        self._sensor_to_rig = None
        if context is not None:
            context.clear_scenes()

    def _timeline(self, recorded_us: Tensor) -> Tensor:
        """Return evenly spaced timestamps to draw the recorded drive along.

        The drive is recorded at whatever rate its logger ran at, which is not
        the rate the model generates at. Poses in between are interpolated, so
        the timeline this returns is the generated rate and the recording is
        read at whatever offsets that lands on.

        Raises:
            ValueError: The drive ends at or before the frame the run continues
                from, leaving nothing to draw.
        """
        step_us = round(1_000_000 / self._frames_per_second)
        # Clamped rather than trusted: the frame a run continues from and the
        # recorded drive are two things the scene has to agree with itself
        # about, and a scene that does not would otherwise draw an empty road.
        start_us = max(int(recorded_us[0].item()), self._view_start_us)
        end_us = int(recorded_us[-1].item())
        if end_us <= start_us:
            raise ValueError(
                f"Scene {self._scene_path} records a drive up to {end_us}us, "
                f"which is not past {start_us}us, where the frame the run "
                "continues from was captured. There is nothing to draw."
            )
        return torch.arange(start_us, end_us + 1, step_us, dtype=torch.int64)

    def _camera_poses(self, timestamps: Tensor) -> Tensor:
        """Return where the camera was at each timestamp, as Ludus wants it.

        Ludus draws from a world-to-camera transform, while the recorded drive
        says where the vehicle was; the camera's mounting is what sits between
        them.
        """
        # [T, 1, 4, 4], one pose per timestamp, which Ludus wants as [T, 4, 4].
        rig_to_world = self._ego_tracks.get_transforms_at_timestamp(timestamps)[:, 0]
        camera_to_world = torch.einsum("nij,jk->nik", rig_to_world, self._sensor_to_rig)
        return torch.linalg.inv(camera_to_world)
