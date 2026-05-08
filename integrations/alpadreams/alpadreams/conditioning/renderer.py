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

"""Ludus-based HD map renderer.

This module provides the LudusRenderer class, which wraps the ludus_renderer
library to render HD map scenes for conditioning video generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import ludus_renderer
import torch
from alpadreams.conditioning.world_scenario.data_types import SceneData
from alpadreams.conditioning.world_scenario.ftheta import FThetaCamera
from alpadreams.conditioning.world_scenario.pinhole import PinholeCamera
from ludus_renderer import (
    load_clipgt_scene,
    mirror_augment_scene,
)
from ludus_renderer.render_utils import (
    SceneAdapter,
)
from ludus_renderer.torch import (
    LudusCudaTimestampedContext,
)
from ludus_renderer.torch.ops import (
    CAMERA_TYPE_REGULAR,
)


class LudusRenderer:
    @staticmethod
    def to_ludus_camera(
        camera: PinholeCamera | FThetaCamera,
    ) -> ludus_renderer.FThetaCamera:
        """
        Convert an Imaginaire camera to a Ludus camera.
        """

        if isinstance(camera, PinholeCamera):
            raise NotImplementedError("Pinhole camera not supported yet")

        elif isinstance(camera, FThetaCamera):
            return ludus_renderer.FThetaCamera(
                principal_point=camera._center_torch,
                image_size=torch.tensor(
                    [camera._width, camera._height], device=camera.device
                ),
                fw_poly=camera._fw_poly_torch,
                max_ray_angle=float(camera._max_ray_angle_torch),
                # linear_distortion=camera._A_torch,  # TODO: Is it _A_torch or _inv_A_torch?
                depth_max=200.0,  # TODO: How to get it from camera data?
            )

        else:
            raise ValueError(f"Unsupported camera type: {type(camera)}")

    def to_ludus_camera_pose(self, camera_poses: torch.Tensor) -> torch.Tensor:
        """
        Convert an Imaginaire camera pose to a Ludus camera pose.
        Args:
            camera_poses: Camera poses [num_frames, 4, 4].
        Returns:
            Ludus camera poses [num_frames, 4, 4].
        """
        return torch.linalg.inv(camera_poses)

    def __init__(
        self,
        scene_data: SceneData,
        camera_models: dict,
        hdmap_color_version: str = "v3",
        bbox_color_version: str = "v3",
        traffic_light_color_version: str = "v2",
        windowless: bool = True,
        device: torch.device = torch.device("cuda"),
        coordinate_system: Literal["FLU", "RDF"] = "FLU",
    ):
        """Render a full sequence for one or more cameras.

        Args:
            args: Command line arguments
            all_cameras: If True, render all available cameras. Otherwise uses args.camera.
        """
        assert hdmap_color_version == "v3", (
            "Only v3 color version is supported for LudusRenderer"
        )
        assert bbox_color_version == "v3", (
            "Only v3 color version is supported for LudusRenderer"
        )
        assert traffic_light_color_version == "v2", (
            "Only v2 color version is supported for LudusRenderer"
        )

        assert len(camera_models) > 0, "Must provide at least one camera model"
        self.scene_data = scene_data
        self.camera_models = camera_models
        self.device = device
        assert coordinate_system == "FLU", (
            "FLU coordinate system is expected for LudusRenderer"
        )

        # Create context
        self.ctx = LudusCudaTimestampedContext(device=self.device)
        self.ctx.set_depth_scaling(True)
        self.ctx.set_msaa_samples(4)
        self.ctx.set_max_tessellation_levels(cube=0)

        # Create and upload cameras
        all_camera_map = {}
        all_cameras = []
        for camera_name, camera_model in camera_models.items():
            cam = self.to_ludus_camera(camera_model)
            all_cameras.append(cam)
            all_camera_map[camera_name] = len(all_cameras) - 1
        self.all_cameras = all_cameras
        self.all_camera_map = all_camera_map
        self.ctx.upload_cameras(all_cameras)

        # Upload scene
        assert "ludus_scene" in self.scene_data.metadata, (
            "Ludus scene not found in scene data"
        )
        scene = self.scene_data.metadata["ludus_scene"]
        self.scene_id = self.ctx.upload_scene(scene.timestamped_scene)

    def render_all_frames_and_cameras(
        self,
        camera_names: list[str],
        camera_poses_per_camera: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        object_infos: list[dict | None] | None = None,
    ) -> torch.Tensor:
        """Render a batch of frames and cameras.

        Args:
            camera_names: List of camera names to render.
            camera_poses_per_camera: Dictionary of camera poses per camera.
            frame_timestamps_us: List of frame timestamps in microseconds.
            object_infos: List of object infos.
        """

        n_cameras = len(camera_names)
        assert n_cameras > 0, "Number of cameras must be greater than 0"

        n_frames = len(frame_timestamps_us)
        assert n_frames > 0, "Number of frames must be greater than 0"

        # Create batch tensors
        scene_id_batch = torch.full(
            (n_frames * n_cameras,),
            self.scene_id,
            dtype=torch.int32,
            device=self.device,
        )
        camera_type_id_batch = torch.full(
            (n_frames * n_cameras,),
            CAMERA_TYPE_REGULAR,
            dtype=torch.int32,
            device=self.device,
        )
        timestamps_batch = torch.tensor(
            frame_timestamps_us, dtype=torch.int64, device=self.device
        ).repeat(n_cameras)

        H, W = None, None

        camera_id_batch = []
        camera_poses_batch = []

        for camera_name in camera_names:
            # Get camera ID, model and check resolution
            c = self.all_camera_map[camera_name]
            m = self.all_cameras[c]
            if H is None or W is None:
                H, W = m.image_size[1], m.image_size[0]
            assert H == m.image_size[1] and W == m.image_size[0], (
                "All cameras must have the same resolution"
            )

            # Append camera ID
            camera_id_batch.append(c)

            # Get camera poses and check shape
            poses = camera_poses_per_camera[camera_name]
            assert len(poses) == n_frames, (
                "Camera poses must have the same length as frame timestamps"
            )
            assert poses.ndim == 3, "Camera poses must be a 3D array"
            assert poses.shape == (n_frames, 4, 4), (
                "Camera poses must have the same length as frame timestamps"
            )
            camera_poses_batch.append(self.to_ludus_camera_pose(poses))

        camera_id_batch = (
            torch.tensor(camera_id_batch, dtype=torch.int32, device=self.device)
            .unsqueeze(1)
            .repeat(1, n_frames)
            .flatten()
        )
        camera_poses_batch = torch.stack(camera_poses_batch, dim=0).reshape(-1, 4, 4)

        images = self.ctx.render(
            scene_id_batch,
            camera_id_batch,
            timestamps_batch,
            camera_type_id_batch,
            camera_poses_batch,
            resolution=(H, W),
        )

        rgb = images[:, :, :, :3]
        if self.ctx.needs_vflip:
            rgb = rgb.flip(1)
        return (
            rgb.squeeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(n_cameras, n_frames, 3, H, W)
        )

    def cleanup(self) -> None:
        """Cleanup the renderer."""
        # LudusCudaTimestampedContext handles cleanup in its destructor
        pass


def load_and_attach_ludus_scene(
    scene_data_path: str | Path,
    scene_data: SceneData,
    device: torch.device = torch.device("cuda"),
    include_ego_trajectory: bool = False,
    include_ego_obstacle: bool = False,
    simplify_dual_lane_lines: bool = False,
    perform_mirror_augment: bool = False,
    n_mirrors: int = 2,
    lookahead_m: float = 50.0,
) -> SceneData:
    """Load HDMap scene from path and attach it to scene data."""
    ludus_scene = load_clipgt_scene(
        scene_data_path,
        device=torch.device(device),
        include_ego_trajectory=include_ego_trajectory,
        include_ego_obstacle=include_ego_obstacle,
        simplify_dual_lane_lines=simplify_dual_lane_lines,
    )
    if perform_mirror_augment:
        augmented_scene = mirror_augment_scene(
            ludus_scene, n_mirrors=n_mirrors, lookahead_m=lookahead_m
        )
    else:
        augmented_scene = ludus_scene
    scene_data.metadata["ludus_scene"] = SceneAdapter(augmented_scene)
    return scene_data
