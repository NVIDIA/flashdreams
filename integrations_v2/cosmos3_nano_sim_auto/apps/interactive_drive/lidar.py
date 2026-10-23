# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrated HD-map LiDAR controls along Interactive Drive's supplied path."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from cosmos_interactive._lidar_codec import network_config
from cosmos_interactive.lidar import LidarGeometry, LidarTokenizer
from torch import Tensor

from cosmos3_nano_sim_auto.impl.joint import stage_timer
from cosmos3_nano_sim_auto.impl.lidar import LidarAssets, LidarSample, LidarStream
from cosmos3_nano_sim_auto.inputs import _object_device
from flashdreams.core.distributed.parallel import ParallelContext

SWEEPS_PER_CHUNK = 4
"""Completed 10 Hz sweeps in one steady 0.4 s chunk: at most this many renderers."""


def lidar_ranks(parallel: ParallelContext | None, num_views: int) -> list[int]:
    """Ranks that render a chunk's sweeps; the first also encodes them.

    Cameras go round-robin from rank zero, so ranks past the rig own none and
    are idle while the cameras draw and encode. Those go first, then the
    camera ranks from the highest (fewest cameras) down. One sweep per rank.
    """
    world = 1 if parallel is None else parallel.world_size
    idle = list(range(num_views, world))
    busy = list(range(min(num_views, world) - 1, -1, -1))
    return (idle + busy)[: min(world, SWEEPS_PER_CHUNK)]


@lru_cache(maxsize=1)
def _codec(
    checkpoint: str,
    device: str,
    version: Literal["v1p1", "v1p2"],
    contiguous_linears: bool,
) -> LidarTokenizer:
    """Codec weights, loaded once per process and reused across restarts."""
    return LidarTokenizer.from_checkpoint(
        checkpoint,
        device=device,
        version=version,
        contiguous_linears=contiguous_linears,
    )


def _geometry_only(version: str) -> LidarTokenizer:
    """Patch geometry without weights, for ranks that neither encode nor decode."""
    config = network_config(version)
    spatial = 2 ** len(config["depths"])
    return LidarTokenizer(
        None,
        LidarGeometry(
            height=config["resolution"][0] // spatial,
            width=config["resolution"][1] // spatial,
        ),
    )


_RENDERER: list[Any] = []
"""``[prepared, settings, renderer]`` for the last scene: a restart reuses its
device points instead of re-uploading the map."""


def _not_a_renderer(*args: Any) -> Tensor:
    raise RuntimeError("This rank renders no LiDAR sweeps")


class _LiveLidarClock(LidarStream):
    def sweep_times(self, first_sweep: int, count: int, device: Any) -> Tensor:
        """Return the unbounded completed-scan clock in seconds."""
        return (
            torch.arange(
                first_sweep, first_sweep + count, device=device, dtype=torch.float64
            )
            / 10.0
        )


class LiveLidarStream(_LiveLidarClock):
    """Render completed 10 Hz sweeps from a bounded, supplied 30 Hz rig path.

    Sweep zero is an instantaneous opening scan. Later sweeps cover the trailing
    interval ending at their timestamp, so rolling shutter never reads an ego
    pose from a future command. The renderer receives both endpoint
    poses and solves each return's firing time within that interval.
    """

    def __init__(
        self,
        assets: LidarAssets,
        *,
        sensor_to_rig: np.ndarray,
        render_sweep: Callable[[int, np.ndarray, np.ndarray], Tensor],
        parallel: ParallelContext | None = None,
        renderers: list[int] | None = None,
    ) -> None:
        super().__init__(assets)
        self._sensor_to_rig = np.asarray(sensor_to_rig, dtype=np.float32)
        self._render_sweep = render_sweep
        self._poses = np.empty((0, 4, 4), dtype=np.float32)
        self._first_frame = 0
        self._supplied = 0
        self._control_presenter = _LiveLidarClock(assets)
        self.last_controls: Tensor | None = None
        self.last_control_range: tuple[int, int] | None = None
        self.last_metrics: dict[str, float] = {}
        self._parallel = parallel if parallel and parallel.world_size > 1 else None
        self._renderers = renderers or [0]
        self._prefetched: tuple[int, int, Tensor | None, Tensor | None] | None = None

    def supply(self, rig_poses_world: np.ndarray) -> None:
        """Append ``[T,4,4]`` host rig poses before generating their sweeps."""
        poses = np.asarray(rig_poses_world, dtype=np.float32)
        if (
            poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or not len(poses)
            or not np.isfinite(poses).all()
            or not np.allclose(poses[:, 3], [0, 0, 0, 1])
            or not np.allclose(
                poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3],
                np.eye(3),
                atol=1e-4,
            )
            or not np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-4)
        ):
            raise ValueError("LiDAR needs a nonempty [T,4,4] finite rigid rig path")
        self._poses = np.concatenate((self._poses, poses.copy()))
        self._supplied += len(poses)

    @property
    def _rank(self) -> int:
        return 0 if self._parallel is None else self._parallel.rank

    def _check(self, start: int, end: int) -> None:
        if start != self.next_sweep or not start < end:
            raise ValueError("Live LiDAR sweeps must be consumed once, consecutively")
        if (end - 1) * 3 >= self._supplied:
            raise ValueError("Supply the driven rig path before rendering LiDAR sweeps")

    def _sweep(self, index: int) -> Tensor:
        first = max(0, (index - 1) * 3) - self._first_frame
        last = index * 3 - self._first_frame
        return self._render_sweep(
            index,
            self._poses[first] @ self._sensor_to_rig,
            self._poses[last] @ self._sensor_to_rig,
        )

    def prefetch(self, start: int, end: int) -> None:
        """Render this rank's sweeps, gather them on the owner and encode there.

        Called by the joint rollout before the camera controls, so the LiDAR
        ranks work while the camera ranks draw. Sweep ``k`` of the chunk is
        rendered by ``renderers[k % len(renderers)]`` and sent point to point to
        the owner (``renderers[0]``), so camera ranks are never held up here.
        Only the owner keeps codec state, so it stays consistent.
        """
        self._check(start, end)
        if self._rank not in self._renderers or (
            self._prefetched is not None and self._prefetched[:2] == (start, end)
        ):
            return
        owner, count = self._renderers[0], len(self._renderers)
        device = self.assets.sample.controls.device
        with stage_timer(device, "lidar/control_render") as render_elapsed:
            mine = {
                index: self._sweep(index)
                for index in range(start, end)
                if self._renderers[(index - start) % count] == self._rank
            }
        metric = result = None
        with stage_timer(device, "lidar/codec_encode") as encode_elapsed:
            if self._parallel is not None and count > 1:
                staging = _object_device(self._parallel)
                rows, columns = self.assets.sample.ray_directions.shape[:2]
                for index in range(start, end):
                    src = self._renderers[(index - start) % count]
                    if src == owner:
                        continue
                    if self._rank == src:
                        dist.send(mine[index].to(staging, torch.float32), dst=owner)
                    elif self._rank == owner:
                        buffer = torch.empty(
                            (3, rows, columns), dtype=torch.float32, device=staging
                        )
                        dist.recv(buffer, src=src)
                        mine[index] = buffer.to(device)
            if self._rank == owner:
                metric = torch.stack([mine[i] for i in range(start, end)])
                metric = metric.permute(1, 0, 2, 3)[None]
                result = self.assets.tokenizer.patchify(self.encoder(metric))
        self.last_metrics = {
            "lidar_control_render_s": render_elapsed(),
            "lidar_codec_encode_s": encode_elapsed(),
        }
        self._prefetched = (start, end, metric, result)

    def tokens(self, start: int, end: int) -> Tensor:
        """Encode consecutive sweeps using only poses already supplied by the app.

        On a mesh the owner broadcasts the tokens, and the metric sweeps rank 0
        presents, so every rank conditions on bitwise-identical controls.
        """
        self._check(start, end)
        self.prefetch(start, end)
        _, _, metric, result = self._prefetched or (0, 0, None, None)
        self._prefetched = None
        if self._parallel is not None:
            g = self.assets.tokenizer.geometry
            staging = _object_device(self._parallel)
            rows, columns = self.assets.sample.ray_directions.shape[:2]
            owner = self._renderers[0]
            if result is not None:
                # One extra row carries the owner's render/encode seconds, so
                # rank 0's chunk line reports them without another collective.
                timings = result.new_zeros(1, g.patch_dim)
                timings[0, 0] = self.last_metrics["lidar_control_render_s"]
                timings[0, 1] = self.last_metrics["lidar_codec_encode_s"]
                result = torch.cat((result, timings))
            received = []
            # ponytail: metric sweeps reach every rank though only rank 0
            # presents them (~11 MB a chunk); send/recv to rank 0 if it shows.
            for mine, shape in (
                (metric, (1, 3, end - start, rows, columns)),
                (result, ((end - start) * g.spatial_tokens + 1, g.patch_dim)),
            ):
                buffer = (
                    torch.empty(shape, dtype=torch.float32, device=staging)
                    if mine is None
                    else mine.to(staging, torch.float32).contiguous()
                )
                dist.broadcast(buffer, src=owner)
                received.append(buffer.to(self._parallel.device))
            metric, result = received
            render_s, encode_s = result[-1, :2].tolist()
            result = result[:-1]
            if self._rank != owner:
                self.last_metrics = {
                    "lidar_control_render_s": render_s,
                    "lidar_codec_encode_s": encode_s,
                }
        assert metric is not None and result is not None
        self.last_controls = metric
        self.last_control_range = (start, end)
        self.next_sweep = end
        # Retain the final endpoint: it starts the next completed scan.
        last_frame = (end - 1) * 3
        self._poses = self._poses[last_frame - self._first_frame :].copy()
        self._first_frame = last_frame
        return result

    def start_decode(self, latent: Tensor) -> Callable[[], Tensor] | None:
        """Decode a chunk's generated LiDAR on the owner and deliver it to rank 0.

        Called by every rank right after the step. The owner (the only rank
        holding codec state) decodes and sends ``[1,3,T,H,W]`` metric ranges,
        plus its decode seconds, to rank 0, so the decode overlaps rank 0's
        camera decode instead of following it. Rank 0 gets a callable that
        waits for the ranges; every other rank gets ``None``.
        """
        owner = self._renderers[0]
        device = self.assets.sample.controls.device
        if self._rank == owner:
            with stage_timer(device, "lidar/codec_decode") as elapsed:
                ranges = self.decoder(latent)
            self.last_metrics["lidar_codec_decode_s"] = elapsed()
            if self._parallel is None or owner == 0:
                return lambda: ranges
            staging = _object_device(self._parallel)
            packed = torch.cat((ranges.flatten(), ranges.new_tensor([elapsed()]))).to(
                staging, torch.float32
            )
            dist.send(packed, dst=0)
            return None
        if self._rank != 0:
            return None
        mesh = self._parallel
        assert mesh is not None
        rows, columns = self.assets.sample.ray_directions.shape[:2]
        shape = (1, 3, int(latent.shape[2]), rows, columns)
        buffer = torch.empty(
            math.prod(shape) + 1,
            dtype=torch.float32,
            device=_object_device(mesh),
        )
        work = dist.irecv(buffer, src=owner)
        assert work is not None

        def received() -> Tensor:
            work.wait()
            self.last_metrics["lidar_codec_decode_s"] = float(buffer[-1])
            return buffer[:-1].view(shape).to(mesh.device)

        return received

    def control_frames(self, **kwargs: Any) -> Tensor:
        """Present the last rendered controls with an independent held sweep."""
        if self.last_controls is None or self.last_control_range is None:
            raise RuntimeError("Render LiDAR controls before presenting them")
        return self._control_presenter.frames(
            self.last_controls, first_sweep=self.last_control_range[0], **kwargs
        )


def make_live_lidar(
    scene: Any,
    *,
    checkpoint: str | Path,
    patch_dim: int,
    device: torch.device | str,
    version: Literal["v1p1", "v1p2"] = "v1p2",
    fps: float = 30.0,
    optimized: bool = False,
    cull: bool = False,
    tf32: bool = False,
    contiguous_linears: bool = False,
    parallel: ParallelContext | None = None,
) -> LiveLidarStream:
    """Prepare the map renderer and trained codec for one live scene.

    Prepared v2 scenes use only local code and their explicit sensor profile.
    """
    if fps != 30.0:
        raise ValueError("Interactive RGB + LiDAR currently requires 30 fps")
    prepared = scene.metadata.get("prepared_scene")
    if prepared is not None:
        from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.prepared_lidar import (
            Renderer,
            require_implementation,
        )

        if prepared.lidar is None:
            raise ValueError(
                "LiDAR requires a v2 prepared scene; use the LiDAR scene ZIP"
            )
        require_implementation()
        gpu = torch.device(device)
        single = parallel is None or parallel.world_size == 1
        renderers = None if single else lidar_ranks(parallel, len(scene.cameras))
        rank = 0 if single else parallel.rank  # type: ignore[union-attr]
        # Each rank builds only what its role uses: renderers the map points on
        # the GPU, the owner (encode and decode) the codec weights.
        if renderers is None or rank in renderers:
            key = (gpu, optimized, cull)
            if not _RENDERER or _RENDERER[0] is not prepared or _RENDERER[1] != key:
                _RENDERER[:] = [
                    prepared,
                    key,
                    Renderer(prepared, device=gpu, optimized=optimized, cull=cull),
                ]
            render_sweep = _RENDERER[2]
        else:
            render_sweep = _not_a_renderer
        opening = int(scene.metadata["pose_frame"])
        sensor_to_rig = np.linalg.inv(
            scene.initial_rig_to_world
        ) @ prepared.lidar.sensor_to_world(prepared, opening)
        tokenizer = (
            _codec(str(checkpoint), str(gpu), version, contiguous_linears)
            if renderers is None or rank == renderers[0]
            else _geometry_only(version)
        ).with_patch_dim(patch_dim)
        sample = LidarSample(
            controls=torch.empty(0, 3, 128, 1800, device=gpu),
            timestamps=torch.empty(0, device=gpu, dtype=torch.float64),
            ray_directions=torch.as_tensor(
                prepared.lidar.model.ray_directions(), device=gpu, dtype=torch.float32
            ),
            sensor_to_ego=torch.as_tensor(
                prepared.lidar.sensor_to_ego, device=gpu, dtype=torch.float32
            ),
        )
        stream = LiveLidarStream(
            LidarAssets(tokenizer, sample),
            sensor_to_rig=sensor_to_rig,
            render_sweep=render_sweep,
            parallel=parallel,
            renderers=renderers,
        )
        stream.encoder.tf32 = stream.decoder.tf32 = tf32
        return stream
    raise ValueError("LiDAR requires a prepared v2 scene bundle")
