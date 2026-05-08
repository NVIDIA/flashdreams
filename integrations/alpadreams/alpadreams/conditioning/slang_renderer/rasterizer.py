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

"""SlangPy-backed HD-map rasterizer with PyTorch tensor I/O.

This module is a flashdreams-tailored adaptation of
``roaddreams.rasterizer.SlangConditionRasterizer``. It differs from the
upstream sample in three substantive ways:

1. **Torch tensors at the boundary.** Camera poses come in as torch CUDA
   tensors and rendered RGB frames are returned as torch CUDA tensors. The
   SlangPy device shares PyTorch's CUDA context via
   :func:`slangpy.create_torch_device` so the inter-op cost is a single
   device-local copy per pass rather than a host round-trip.

2. **Multi-camera batching.** :meth:`SlangConditionRasterizer.render_camera`
   loops over the per-camera frame batch internally, returning ``[N, 3, H, W]``
   on GPU. The per-camera projection state (LUTs, linear matrix, principal
   point) is rebuilt once at construction and reused across frames.

3. **Static SceneBundle input.** The Slang kernels treat the scene as static;
   ``timestamps_us`` is accepted to match the LudusRenderer signature but is
   currently unused (flashdreams loads HD-map-only conditioning scenes).
"""

from __future__ import annotations

from contextlib import nullcontext
import importlib.resources
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from alpadreams.conditioning.slang_renderer.camera import FThetaCameraModel
from alpadreams.conditioning.slang_renderer.config import RasterConfig
from alpadreams.conditioning.slang_renderer.patterns import triangulate_polygon_xy
from alpadreams.conditioning.slang_renderer.types import SceneBundle


class SlangConditionRasterizer:
    """Renders HD-map conditioning frames from a static SceneBundle.

    One instance owns the SlangPy device, the compiled module, scene buffers,
    and per-camera projection state. Construct it once per ``LudusRenderer``,
    upload a scene with :meth:`load_scene`, then render with
    :meth:`render_camera` for each camera in your batch.
    """

    def __init__(
        self,
        camera_models: dict[str, FThetaCameraModel],
        raster: RasterConfig,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(f"SlangConditionRasterizer requires a CUDA device, got {device}")
        if not camera_models:
            raise ValueError("camera_models must contain at least one entry")

        try:
            import slangpy as spy
            from slangpy import Module
        except ImportError as exc:  # pragma: no cover - environment misconfig
            raise RuntimeError(
                "SlangPy is required for the conditioning renderer. "
                "Install with `uv sync` so `slangpy` is resolved."
            ) from exc

        self._spy = spy
        self._raster = raster
        self._torch_device = device
        self._camera_models = dict(camera_models)

        for name, model in self._camera_models.items():
            if model.output_width != raster.width or model.output_height != raster.height:
                raise ValueError(
                    f"Camera '{name}' resolution ({model.output_width}x{model.output_height}) "
                    f"does not match RasterConfig ({raster.width}x{raster.height})"
                )

        shader_dir = importlib.resources.files(
            "alpadreams.conditioning.slang_renderer"
        ).joinpath("slang")
        # slangpy.create_torch_device automatically prepends slangpy's own shader
        # path; we only need to add our own kernel directory.
        self._device = spy.create_torch_device(
            type=spy.DeviceType.cuda,
            torch_device=device,
            include_paths=[str(shader_dir)],
        )
        self._module = Module(self._device.load_module("rasterizer.slang"))

        self._dummy_u32_buffer = self._device.create_buffer(
            usage=(
                spy.BufferUsage.shader_resource
                | spy.BufferUsage.unordered_access
                | spy.BufferUsage.copy_source
                | spy.BufferUsage.copy_destination
            ),
            label="dummy_u32",
            data=np.zeros((1,), dtype=np.uint32),
        )

        self._rgb_tensor = spy.Tensor.empty(
            self._device, dtype=self._module.float4, shape=(raster.height, raster.width)
        )
        self._depth_tensor = spy.Tensor.empty(
            self._device, dtype=self._module.float, shape=(raster.height, raster.width)
        )
        self._depth_bits_tensor = spy.Tensor.empty(
            self._device, dtype=self._module.uint, shape=(raster.height, raster.width)
        )
        self._rgb_rgba8_tensor = spy.Tensor.empty(
            self._device, dtype=self._module.uint, shape=(raster.height, raster.width)
        )

        self._scene: SceneBundle | None = None
        self._line_buffer: Any | None = None
        self._line_count: int = 0
        self._scene_triangle_buffer: Any | None = None
        self._scene_triangle_count: int = 0
        self._polygon_triangle_buffer: Any | None = None
        self._polygon_triangle_count: int = 0
        self._near_triangle_mask_buffer: Any | None = None
        self._near_triangle_index_buffer: Any | None = None
        self._near_triangle_counter_buffer: Any | None = None

        self._camera_state: dict[str, _CameraState] = {}
        for name, model in self._camera_models.items():
            self._camera_state[name] = self._build_camera_state(model)

    def load_scene(self, scene: SceneBundle) -> None:
        """Upload a SceneBundle's geometry into GPU buffers."""
        with self._device_context():
            self._scene = scene
            self._line_buffer, self._line_count = self._upload_line_primitives(scene)
            (
                self._scene_triangle_buffer,
                self._scene_triangle_count,
                self._polygon_triangle_buffer,
                self._polygon_triangle_count,
            ) = self._upload_triangle_primitives(scene)

            capacity = self._polygon_triangle_count
            if capacity > 0:
                self._near_triangle_mask_buffer = self._create_u32_buffer(
                    element_count=capacity, label="near_triangle_mask"
                )
                self._near_triangle_index_buffer = self._create_u32_buffer(
                    element_count=capacity, label="near_triangle_indices"
                )
                self._near_triangle_counter_buffer = self._create_u32_buffer(
                    element_count=4, label="near_triangle_counter"
                )
            else:
                self._near_triangle_mask_buffer = None
                self._near_triangle_index_buffer = None
                self._near_triangle_counter_buffer = None

    def render_camera(
        self,
        camera_name: str,
        camera_to_world: torch.Tensor,
    ) -> torch.Tensor:
        """Render a sequence of frames for a single camera.

        Args:
            camera_name: Name registered in ``camera_models`` at construction.
            camera_to_world: ``[N, 4, 4]`` camera-to-world poses in the same
                world frame as the loaded SceneBundle (flashdreams' CLIPGT
                loader emits OpenCV RDF; pass ego poses straight through).
                On the same CUDA device as this rasterizer.

        Returns:
            ``[N, 3, H, W]`` uint8 RGB tensor on the rasterizer's CUDA device.
        """
        if self._scene is None:
            raise RuntimeError("load_scene() must be called before render_camera()")
        if camera_name not in self._camera_state:
            raise KeyError(f"Camera '{camera_name}' was not registered at construction time")
        if camera_to_world.ndim != 3 or camera_to_world.shape[-2:] != (4, 4):
            raise ValueError(
                f"camera_to_world must be [N, 4, 4], got shape {tuple(camera_to_world.shape)}"
            )

        n_frames = int(camera_to_world.shape[0])
        H = self._raster.height
        W = self._raster.width
        out = torch.empty((n_frames, 3, H, W), dtype=torch.uint8, device=self._torch_device)
        if n_frames == 0:
            return out

        camera_state = self._camera_state[camera_name]
        poses_np = camera_to_world.detach().to(torch.float32).cpu().numpy()
        for frame_idx in range(n_frames):
            world_to_camera = np.linalg.inv(poses_np[frame_idx]).astype(np.float32)
            self._render_single_frame(camera_state, world_to_camera, out[frame_idx])
        return out

    def _render_single_frame(
        self,
        camera: "_CameraState",
        world_to_camera_rdf: npt.NDArray[np.float32],
        out_chw: torch.Tensor,
    ) -> None:
        with self._device_context():
            spy = self._spy
            width = self._raster.width
            height = self._raster.height

            world_to_camera_uniform = spy.float4x4(world_to_camera_rdf.reshape(-1).tolist())
            principal_uniform = spy.float2(float(camera.principal_px[0]), float(camera.principal_px[1]))
            linear_row0_uniform = spy.float2(float(camera.linear_row0[0]), float(camera.linear_row0[1]))
            linear_row1_uniform = spy.float2(float(camera.linear_row1[0]), float(camera.linear_row1[1]))
            inv_linear_row0_uniform = spy.float2(
                float(camera.inv_linear_row0[0]), float(camera.inv_linear_row0[1])
            )
            inv_linear_row1_uniform = spy.float2(
                float(camera.inv_linear_row1[0]), float(camera.inv_linear_row1[1])
            )

            near_triangle_count = 0
            if (
                self._polygon_triangle_count > 0
                and self._polygon_triangle_buffer is not None
                and self._near_triangle_mask_buffer is not None
                and self._near_triangle_index_buffer is not None
                and self._near_triangle_counter_buffer is not None
            ):
                classify_encoder = self._device.create_command_encoder()
                classify_encoder.clear_buffer(self._near_triangle_mask_buffer)
                classify_encoder.clear_buffer(self._near_triangle_counter_buffer)
                self._module.classify_near_triangles_world.dispatch(
                    spy.uint3(self._polygon_triangle_count, 1, 1),
                    triangles=self._polygon_triangle_buffer,
                    near_triangle_mask=self._near_triangle_mask_buffer,
                    near_triangle_indices=self._near_triangle_index_buffer,
                    counters=self._near_triangle_counter_buffer,
                    world_to_camera_rdf=world_to_camera_uniform,
                    triangle_raytrace_distance_m=float(self._raster.triangle_raytrace_distance_m),
                    command_encoder=classify_encoder,
                )
                self._device.submit_command_buffer(classify_encoder.finish())
                near_triangle_count = int(self._read_u32_buffer(self._near_triangle_counter_buffer, count=1)[0])

            clear_encoder = self._device.create_command_encoder()
            self._module.clear_targets.dispatch(
                spy.uint3(width, height, 1),
                rgb=self._rgb_tensor,
                depth=self._depth_tensor,
                depth_bits=self._depth_bits_tensor,
                image_width=width,
                image_height=height,
                clear_depth=float(self._raster.depth_clear_m),
                command_encoder=clear_encoder,
            )
            self._device.submit_command_buffer(clear_encoder.finish())

            if self._line_count > 0 and self._line_buffer is not None:
                lines_encoder = self._device.create_command_encoder()
                self._module.rasterize_lines_world.dispatch(
                    spy.uint3(self._line_count, 1, 1),
                    lines=self._line_buffer,
                    angle_to_radius_lut=camera.angle_to_radius_lut,
                    rgb=self._rgb_tensor,
                    depth=self._depth_tensor,
                    depth_bits=self._depth_bits_tensor,
                    image_width=width,
                    image_height=height,
                    world_to_camera_rdf=world_to_camera_uniform,
                    principal_px=principal_uniform,
                    linear_row0=linear_row0_uniform,
                    linear_row1=linear_row1_uniform,
                    near_plane_m=float(self._raster.near_plane_m),
                    lut_count=camera.angle_lut_count,
                    lut_max_angle=float(camera.angle_lut_max_angle),
                    lut_max_radius=float(camera.angle_lut_max_radius),
                    lut_tail_slope=float(camera.angle_lut_tail_slope),
                    fog_start_m=float(self._raster.fog_start_m),
                    fog_end_m=float(self._raster.fog_end_m),
                    fog_power=float(self._raster.fog_power),
                    command_encoder=lines_encoder,
                )
                self._device.submit_command_buffer(lines_encoder.finish())

            if self._scene_triangle_count > 0 and self._scene_triangle_buffer is not None:
                scene_tri_encoder = self._device.create_command_encoder()
                self._module.rasterize_triangles_world.dispatch(
                    spy.uint3(self._scene_triangle_count, 1, 1),
                    triangles=self._scene_triangle_buffer,
                    near_triangle_mask=self._near_triangle_mask_buffer or self._dummy_u32_buffer,
                    angle_to_radius_lut=camera.angle_to_radius_lut,
                    rgb=self._rgb_tensor,
                    depth=self._depth_tensor,
                    depth_bits=self._depth_bits_tensor,
                    image_width=width,
                    image_height=height,
                    world_to_camera_rdf=world_to_camera_uniform,
                    principal_px=principal_uniform,
                    linear_row0=linear_row0_uniform,
                    linear_row1=linear_row1_uniform,
                    near_plane_m=float(self._raster.near_plane_m),
                    lut_count=camera.angle_lut_count,
                    lut_max_angle=float(camera.angle_lut_max_angle),
                    lut_max_radius=float(camera.angle_lut_max_radius),
                    lut_tail_slope=float(camera.angle_lut_tail_slope),
                    skip_masked=0,
                    fog_start_m=float(self._raster.fog_start_m),
                    fog_end_m=float(self._raster.fog_end_m),
                    fog_power=float(self._raster.fog_power),
                    command_encoder=scene_tri_encoder,
                )
                self._device.submit_command_buffer(scene_tri_encoder.finish())

            if (
                self._polygon_triangle_count > 0
                and self._polygon_triangle_buffer is not None
                and self._near_triangle_mask_buffer is not None
            ):
                polygon_encoder = self._device.create_command_encoder()
                self._module.rasterize_triangles_world.dispatch(
                    spy.uint3(self._polygon_triangle_count, 1, 1),
                    triangles=self._polygon_triangle_buffer,
                    near_triangle_mask=self._near_triangle_mask_buffer,
                    angle_to_radius_lut=camera.angle_to_radius_lut,
                    rgb=self._rgb_tensor,
                    depth=self._depth_tensor,
                    depth_bits=self._depth_bits_tensor,
                    image_width=width,
                    image_height=height,
                    world_to_camera_rdf=world_to_camera_uniform,
                    principal_px=principal_uniform,
                    linear_row0=linear_row0_uniform,
                    linear_row1=linear_row1_uniform,
                    near_plane_m=float(self._raster.near_plane_m),
                    lut_count=camera.angle_lut_count,
                    lut_max_angle=float(camera.angle_lut_max_angle),
                    lut_max_radius=float(camera.angle_lut_max_radius),
                    lut_tail_slope=float(camera.angle_lut_tail_slope),
                    skip_masked=1,
                    fog_start_m=float(self._raster.fog_start_m),
                    fog_end_m=float(self._raster.fog_end_m),
                    fog_power=float(self._raster.fog_power),
                    command_encoder=polygon_encoder,
                )
                self._device.submit_command_buffer(polygon_encoder.finish())

            if (
                near_triangle_count > 0
                and self._polygon_triangle_buffer is not None
                and self._near_triangle_index_buffer is not None
            ):
                raytrace_encoder = self._device.create_command_encoder()
                self._module.raytrace_triangles_world.dispatch(
                    spy.uint3(width, height, 1),
                    near_triangle_indices=self._near_triangle_index_buffer,
                    triangles=self._polygon_triangle_buffer,
                    radius_to_angle_lut=camera.radius_to_angle_lut,
                    rgb=self._rgb_tensor,
                    depth=self._depth_tensor,
                    depth_bits=self._depth_bits_tensor,
                    image_width=width,
                    image_height=height,
                    world_to_camera_rdf=world_to_camera_uniform,
                    principal_px=principal_uniform,
                    inv_linear_row0=inv_linear_row0_uniform,
                    inv_linear_row1=inv_linear_row1_uniform,
                    near_plane_m=float(self._raster.near_plane_m),
                    radius_lut_count=camera.radius_lut_count,
                    radius_lut_max_radius=float(camera.radius_lut_max_radius),
                    radius_lut_tail_slope=float(camera.radius_lut_tail_slope),
                    near_triangle_count=near_triangle_count,
                    fog_start_m=float(self._raster.fog_start_m),
                    fog_end_m=float(self._raster.fog_end_m),
                    fog_power=float(self._raster.fog_power),
                    command_encoder=raytrace_encoder,
                )
                self._device.submit_command_buffer(raytrace_encoder.finish())

            convert_encoder = self._device.create_command_encoder()
            self._module.pack_rgb_to_rgba8.dispatch(
                spy.uint3(width, height, 1),
                rgb=self._rgb_tensor,
                rgba8=self._rgb_rgba8_tensor,
                image_width=width,
                image_height=height,
                command_encoder=convert_encoder,
            )
            self._device.submit_command_buffer(convert_encoder.finish())

            # The RGBA8 tensor lives on the same CUDA context as PyTorch (we
            # built the SlangPy device via create_torch_device above), so
            # to_torch() returns a zero-copy view of the GPU-resident buffer.
            rgba_packed = self._rgb_rgba8_tensor.to_torch()
            rgba_uint8 = rgba_packed.view(torch.uint8).reshape(height, width, 4)
            rgb_chw = rgba_uint8[..., :3].permute(2, 0, 1)
            out_chw.copy_(rgb_chw)

    def _build_camera_state(self, model: FThetaCameraModel) -> "_CameraState":
        radii, angle_count, max_angle, max_radius, angle_tail_slope = model.build_angle_to_radius_lut()
        angles, radius_count, radius_max_radius, radius_tail_slope = model.build_radius_to_angle_lut()
        spy = self._spy
        angle_buffer = self._device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="angle_to_radius_lut",
            data=np.ascontiguousarray(radii, dtype=np.float32),
        )
        radius_buffer = self._device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            label="radius_to_angle_lut",
            data=np.ascontiguousarray(angles, dtype=np.float32),
        )
        row0, row1, inv_row0, inv_row1 = model.scaled_linear_rows()
        return _CameraState(
            angle_to_radius_lut=angle_buffer,
            angle_lut_count=angle_count,
            angle_lut_max_angle=max_angle,
            angle_lut_max_radius=max_radius,
            angle_lut_tail_slope=angle_tail_slope,
            radius_to_angle_lut=radius_buffer,
            radius_lut_count=radius_count,
            radius_lut_max_radius=radius_max_radius,
            radius_lut_tail_slope=radius_tail_slope,
            principal_px=model.principal_px_scaled(),
            linear_row0=row0,
            linear_row1=row1,
            inv_linear_row0=inv_row0,
            inv_linear_row1=inv_row1,
        )

    def _upload_line_primitives(self, scene: SceneBundle) -> tuple[Any | None, int]:
        line_batches: list[np.ndarray] = []
        for layer in scene.line_layers:
            if len(layer.segments_world) == 0:
                continue
            segments = np.asarray(layer.segments_world, dtype=np.float32)
            if not bool(np.all(np.isfinite(segments))):
                # Last-line-of-defence guard. A single NaN endpoint here
                # poisons the GPU line raster pass: the shader's
                # ``distance_to_line > half_width`` check evaluates to
                # ``false`` for NaN inputs, so every pixel inside the
                # bounding box gets painted, producing the giant blob /
                # speckle pattern. See
                # https://gitlab-master.nvidia.com/sil/omni-dreams/-/commit/92091220
                bad = int(np.sum(~np.all(np.isfinite(segments.reshape(-1, 6)), axis=1)))
                print(
                    f"[rasterizer] WARNING: dropping line layer "
                    f"'{layer.layer_name}' — {bad} of {segments.shape[0]} "
                    "segment(s) contain NaN/Inf. Fix the scene loader.",
                    flush=True,
                )
                continue
            batch = np.zeros((segments.shape[0], 16), dtype=np.float32)
            batch[:, 0:3] = segments[:, 0, :]
            batch[:, 3] = 1.0
            batch[:, 4:7] = segments[:, 1, :]
            batch[:, 7] = 1.0
            batch[:, 8:12] = np.asarray(layer.color_rgba, dtype=np.float32)
            batch[:, 12] = np.float32(layer.width_px)
            line_batches.append(batch)

        if not line_batches:
            return None, 0

        data = np.ascontiguousarray(np.concatenate(line_batches, axis=0), dtype=np.float32)
        return (
            self._device.create_buffer(
                usage=self._spy.BufferUsage.shader_resource,
                label="line_primitives",
                data=data,
            ),
            int(data.shape[0]),
        )

    def _upload_triangle_primitives(
        self, scene: SceneBundle
    ) -> tuple[Any | None, int, Any | None, int]:
        scene_triangle_batches: list[np.ndarray] = []
        scene_triangle_count = 0
        for layer in scene.triangle_layers:
            if len(layer.triangles_world) == 0:
                continue
            triangles = np.asarray(layer.triangles_world, dtype=np.float32)
            if not bool(np.all(np.isfinite(triangles))):
                bad = int(np.sum(~np.all(np.isfinite(triangles.reshape(-1, 9)), axis=1)))
                print(
                    f"[rasterizer] WARNING: dropping triangle layer "
                    f"'{layer.layer_name}' — {bad} of {triangles.shape[0]} "
                    "triangle(s) contain NaN/Inf.",
                    flush=True,
                )
                continue
            scene_triangle_count += int(triangles.shape[0])
            scene_triangle_batches.append(
                self._pack_triangle_batch(triangles, np.asarray(layer.color_rgba, dtype=np.float32))
            )

        polygon_triangle_batches: list[np.ndarray] = []
        polygon_triangle_count = 0
        for layer in scene.polygon_layers:
            color = np.asarray(layer.color_rgba, dtype=np.float32)
            for polygon_world in layer.polygons_world:
                polygon_arr = np.asarray(polygon_world, dtype=np.float32)
                if not bool(np.all(np.isfinite(polygon_arr))):
                    print(
                        f"[rasterizer] WARNING: skipping polygon in layer "
                        f"'{layer.layer_name}' — vertices contain NaN/Inf.",
                        flush=True,
                    )
                    continue
                polygon_triangles = triangulate_polygon_xy(polygon_arr)
                if len(polygon_triangles) == 0 or not bool(np.all(np.isfinite(polygon_triangles))):
                    continue
                polygon_triangle_count += int(len(polygon_triangles))
                polygon_triangle_batches.append(self._pack_triangle_batch(polygon_triangles, color))

        scene_triangle_buffer = None
        if scene_triangle_batches:
            scene_triangle_data = np.ascontiguousarray(np.concatenate(scene_triangle_batches, axis=0), dtype=np.float32)
            scene_triangle_buffer = self._device.create_buffer(
                usage=self._spy.BufferUsage.shader_resource,
                label="scene_triangle_primitives",
                data=scene_triangle_data,
            )

        polygon_triangle_buffer = None
        if polygon_triangle_batches:
            polygon_triangle_data = np.ascontiguousarray(np.concatenate(polygon_triangle_batches, axis=0), dtype=np.float32)
            polygon_triangle_buffer = self._device.create_buffer(
                usage=self._spy.BufferUsage.shader_resource,
                label="polygon_triangle_primitives",
                data=polygon_triangle_data,
            )

        return (
            scene_triangle_buffer,
            scene_triangle_count,
            polygon_triangle_buffer,
            polygon_triangle_count,
        )

    def _pack_triangle_batch(
        self,
        triangles_world: npt.NDArray[np.float32],
        color_rgba: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        triangles = np.asarray(triangles_world, dtype=np.float32)
        if len(triangles) == 0:
            return np.empty((0, 16), dtype=np.float32)
        batch = np.zeros((triangles.shape[0], 16), dtype=np.float32)
        batch[:, 0:3] = triangles[:, 0, :]
        batch[:, 3] = 1.0
        batch[:, 4:7] = triangles[:, 1, :]
        batch[:, 7] = 1.0
        batch[:, 8:11] = triangles[:, 2, :]
        batch[:, 11] = 1.0
        batch[:, 12:16] = np.asarray(color_rgba, dtype=np.float32)
        return batch

    def _create_u32_buffer(self, *, element_count: int, label: str) -> Any:
        return self._device.create_buffer(
            usage=(
                self._spy.BufferUsage.shader_resource
                | self._spy.BufferUsage.unordered_access
                | self._spy.BufferUsage.copy_source
                | self._spy.BufferUsage.copy_destination
            ),
            label=label,
            data=np.zeros((element_count,), dtype=np.uint32),
        )

    def _read_u32_buffer(self, buffer: Any, *, count: int) -> npt.NDArray[np.uint32]:
        raw = np.asarray(buffer.to_numpy(), dtype=np.uint8)
        values = raw.view(np.uint32)
        return values[:count].copy()

    def _device_context(self):
        if hasattr(self._device, "cuda_context_scope"):
            return self._device.cuda_context_scope()
        return nullcontext()


class _CameraState:
    """Per-camera GPU buffers and uniforms used by the Slang shader."""

    __slots__ = (
        "angle_to_radius_lut",
        "angle_lut_count",
        "angle_lut_max_angle",
        "angle_lut_max_radius",
        "angle_lut_tail_slope",
        "radius_to_angle_lut",
        "radius_lut_count",
        "radius_lut_max_radius",
        "radius_lut_tail_slope",
        "principal_px",
        "linear_row0",
        "linear_row1",
        "inv_linear_row0",
        "inv_linear_row1",
    )

    def __init__(
        self,
        *,
        angle_to_radius_lut: Any,
        angle_lut_count: int,
        angle_lut_max_angle: float,
        angle_lut_max_radius: float,
        angle_lut_tail_slope: float,
        radius_to_angle_lut: Any,
        radius_lut_count: int,
        radius_lut_max_radius: float,
        radius_lut_tail_slope: float,
        principal_px: npt.NDArray[np.float32],
        linear_row0: npt.NDArray[np.float32],
        linear_row1: npt.NDArray[np.float32],
        inv_linear_row0: npt.NDArray[np.float32],
        inv_linear_row1: npt.NDArray[np.float32],
    ) -> None:
        self.angle_to_radius_lut = angle_to_radius_lut
        self.angle_lut_count = angle_lut_count
        self.angle_lut_max_angle = angle_lut_max_angle
        self.angle_lut_max_radius = angle_lut_max_radius
        self.angle_lut_tail_slope = angle_lut_tail_slope
        self.radius_to_angle_lut = radius_to_angle_lut
        self.radius_lut_count = radius_lut_count
        self.radius_lut_max_radius = radius_lut_max_radius
        self.radius_lut_tail_slope = radius_lut_tail_slope
        self.principal_px = principal_px
        self.linear_row0 = linear_row0
        self.linear_row1 = linear_row1
        self.inv_linear_row0 = inv_linear_row0
        self.inv_linear_row1 = inv_linear_row1
