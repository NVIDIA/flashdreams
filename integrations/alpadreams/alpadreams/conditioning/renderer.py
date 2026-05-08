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

"""SlangPy-based HD-map renderer.

This module provides the :class:`LudusRenderer` class, which renders HD-map
conditioning frames for the Alpadreams video model. The class name is
preserved for backwards compatibility, but the implementation no longer
depends on the closed-source ``ludus-renderer`` wheel: it now uses a SlangPy
software rasterizer (see :mod:`alpadreams.conditioning.slang_renderer`) that
runs custom Slang compute kernels under PyTorch's CUDA context.

Public API surface (``__init__``, :meth:`render_all_frames_and_cameras`,
:meth:`cleanup`, :func:`load_and_attach_ludus_scene`) is unchanged so callers
in :mod:`alpadreams.grpc` work without modification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from alpadreams.conditioning.slang_renderer import (
    RasterConfig,
    SceneBundle,
    SlangConditionRasterizer,
    mirror_augment_bundle,
    scene_data_to_bundle,
)
from alpadreams.conditioning.slang_renderer.camera import FThetaCameraModel
from alpadreams.conditioning.world_scenario.data_loaders import load_scene
from alpadreams.conditioning.world_scenario.data_types import SceneData
from alpadreams.conditioning.world_scenario.ftheta import FThetaCamera
from alpadreams.conditioning.world_scenario.pinhole import PinholeCamera

SCENE_BUNDLE_KEY = "ludus_scene"


class LudusRenderer:
    """Render HD-map conditioning frames for one or more cameras.

    The class wraps :class:`SlangConditionRasterizer` with the multi-camera
    batching contract that :class:`AlpadreamsConditioningWrapper` expects:
    ``render_all_frames_and_cameras`` returns a ``[V, T, 3, H, W]`` uint8
    tensor on GPU.
    """

    def __init__(
        self,
        scene_data: SceneData,
        camera_models: dict[str, FThetaCamera | PinholeCamera],
        hdmap_color_version: str = "v3",
        bbox_color_version: str = "v3",
        traffic_light_color_version: str = "v2",
        windowless: bool = True,
        device: torch.device = torch.device("cuda"),
        coordinate_system: Literal["FLU", "RDF"] = "FLU",
    ):
        """Construct a renderer for a fixed set of cameras.

        Args:
            scene_data: World scenario data; must already have a SceneBundle
                attached under ``metadata[SCENE_BUNDLE_KEY]`` (see
                :func:`load_and_attach_ludus_scene`).
            camera_models: ``{camera_name: FThetaCamera}`` dictionary. Pinhole
                cameras are not supported.
            hdmap_color_version: Must be ``"v3"`` (other palettes are not
                ported from ludus).
            bbox_color_version: Must be ``"v3"``.
            traffic_light_color_version: Must be ``"v2"``.
            windowless: Unused; kept for API parity.
            device: CUDA device on which the rasterizer runs.
            coordinate_system: Must be ``"FLU"``; the rasterizer converts to
                RDF internally.
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
        assert coordinate_system == "FLU", (
            "FLU coordinate system is expected for LudusRenderer"
        )
        assert len(camera_models) > 0, "Must provide at least one camera model"
        del windowless  # accepted for API compatibility, never used

        self.scene_data = scene_data
        self.camera_models = camera_models
        self.device = device

        bundle = scene_data.metadata.get(SCENE_BUNDLE_KEY)
        if not isinstance(bundle, SceneBundle):
            raise ValueError(
                f"scene_data.metadata['{SCENE_BUNDLE_KEY}'] must be a SceneBundle. "
                "Did you forget to call load_and_attach_ludus_scene()?"
            )

        widths: set[int] = set()
        heights: set[int] = set()
        for camera_name, camera_model in camera_models.items():
            if isinstance(camera_model, PinholeCamera):
                raise NotImplementedError(
                    f"Pinhole camera not supported by SlangPy rasterizer (camera '{camera_name}')."
                )
            if not isinstance(camera_model, FThetaCamera):
                raise TypeError(
                    f"Unsupported camera type for '{camera_name}': {type(camera_model)}"
                )
            widths.add(int(camera_model.width))
            heights.add(int(camera_model.height))

        if len(widths) != 1 or len(heights) != 1:
            raise ValueError(
                f"All cameras must share resolution; got widths={widths}, heights={heights}"
            )
        width = widths.pop()
        height = heights.pop()

        raster_config = RasterConfig(width=width, height=height, compute_device="cuda")
        slang_camera_models: dict[str, FThetaCameraModel] = {
            name: FThetaCameraModel(model, output_width=width, output_height=height)
            for name, model in camera_models.items()
        }

        self._rasterizer = SlangConditionRasterizer(
            camera_models=slang_camera_models,
            raster=raster_config,
            device=self.device,
        )
        self._rasterizer.load_scene(bundle)

    def render_all_frames_and_cameras(
        self,
        camera_names: list[str],
        camera_poses_per_camera: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        object_infos: list[dict | None] | None = None,
    ) -> torch.Tensor:
        """Render a batch of frames for the given cameras.

        Args:
            camera_names: Ordered list of camera names (defines the V axis).
            camera_poses_per_camera: ``{camera_name: [T, 4, 4]}`` camera-to-world
                poses in OpenCV RDF (matching ``scene_data.metadata["coordinate_frame"]``,
                which is always ``opencv_rdf`` for CLIPGT-loaded scenes). The
                gRPC server converts client-supplied FLU poses to RDF inside
                :func:`alpadreams.grpc.utils.compute_camera_poses_from_rig`,
                so callers there get this for free; if you build poses from
                some other source make sure they're already in RDF.
            frame_timestamps_us: Frame timestamps in microseconds (unused; the
                Slang rasterizer treats the scene as static, but we accept the
                argument for API parity with the previous ludus path).
            object_infos: Per-frame dynamic-object info dicts (unused; bbox
                tracks are not yet ported to the SlangPy backend).

        Returns:
            ``[V, T, 3, H, W]`` uint8 tensor on the renderer's CUDA device.
        """
        del object_infos  # bbox tracks are not yet ported

        n_cameras = len(camera_names)
        if n_cameras == 0:
            raise ValueError("camera_names must be non-empty")
        n_frames = len(frame_timestamps_us)
        if n_frames == 0:
            raise ValueError("frame_timestamps_us must be non-empty")

        per_view: list[torch.Tensor] = []
        for camera_name in camera_names:
            if camera_name not in self.camera_models:
                raise KeyError(f"Unknown camera name: {camera_name}")
            poses = camera_poses_per_camera[camera_name]
            if poses.shape != (n_frames, 4, 4):
                raise ValueError(
                    f"camera_poses for '{camera_name}' must be [{n_frames}, 4, 4], got {tuple(poses.shape)}"
                )
            poses_cuda = poses.to(device=self.device, dtype=torch.float32)
            frames = self._rasterizer.render_camera(camera_name, poses_cuda)
            per_view.append(frames)

        return torch.stack(per_view, dim=0).contiguous()

    def cleanup(self) -> None:
        """Release rasterizer resources (no-op for now)."""
        # SlangPy device cleanup is handled by garbage collection.
        return None


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
    """Build a SceneBundle from ``scene_data`` and attach it to its metadata.

    The function name is preserved for backwards compatibility with
    ``alpadreams.grpc.utils.load_static_world_from_zip_bytes``; the SceneBundle
    is stored under ``scene_data.metadata[SCENE_BUNDLE_KEY]`` for the
    LudusRenderer constructor to consume.

    Args:
        scene_data_path: Unused; kept for API parity (the previous
            implementation reloaded the scene from disk via ludus, but
            flashdreams' :class:`SceneData` already carries everything we need).
        scene_data: World scenario data populated by :func:`load_scene`.
        device: Unused; the SceneBundle is held on the host (numpy arrays) and
            uploaded to GPU at render time.
        include_ego_trajectory: Reserved for future ego-trajectory rendering;
            currently a no-op (matches the default flashdreams config).
        include_ego_obstacle: Reserved for future ego-obstacle rendering;
            currently a no-op.
        simplify_dual_lane_lines: Reserved; the new pattern engine already
            applies the standard lane-line simplifications.
        perform_mirror_augment: If True, mirror-stitches the scene
            ``n_mirrors`` times.
        n_mirrors: Number of mirror copies to add.
        lookahead_m: Distance past the last ego pose to place the first mirror plane.

    Returns:
        ``scene_data`` with ``metadata[SCENE_BUNDLE_KEY]`` set to the new bundle.
    """
    del scene_data_path  # not needed: SceneData is already populated
    del device
    del include_ego_trajectory
    del include_ego_obstacle
    del simplify_dual_lane_lines

    raster_config = RasterConfig()
    bundle = scene_data_to_bundle(scene_data, raster_config)
    if perform_mirror_augment:
        bundle = mirror_augment_bundle(
            bundle,
            scene_data.ego_poses,
            n_mirrors=n_mirrors,
            lookahead_m=lookahead_m,
        )
    scene_data.metadata[SCENE_BUNDLE_KEY] = bundle
    return scene_data


__all__ = [
    "LudusRenderer",
    "SCENE_BUNDLE_KEY",
    "load_and_attach_ludus_scene",
]


# Re-exports kept here so existing code that references these names through
# alpadreams.conditioning.renderer keeps working.
_ = (Any, load_scene)
