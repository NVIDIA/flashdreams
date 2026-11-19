# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record synchronized driving frames and state for offline presentation."""

from __future__ import annotations

import json
import shutil
import time
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from secrets import token_hex
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from PIL import Image

from .config import AppConfig
from .types import DriverCommand, FrameChunk, SceneBundle, TrajectoryChunk


class DriveRecording:
    """Append native camera frames and publish a ZIP only after finalization."""

    def __init__(self, root: Path, config: AppConfig, scene: SceneBundle) -> None:
        self.id = time.strftime("drive-%Y%m%d-%H%M%S-", time.gmtime()) + token_hex(4)
        self.path = root.resolve() / self.id
        self.path.mkdir(parents=True)
        self.frames = 0
        self._start_us: int | None = None
        self._last_us: int | None = None
        self._begin_us: int | None = None
        self._pending: dict[int, dict[str, Any]] = {}
        self._closed = False
        self._cameras = config.camera_names
        self._records = (self.path / "frames.jsonl").open("x")
        self._commands = (self.path / "commands.jsonl").open("x")
        self._manifest = {
            "format": "flashdreams.drive-recording",
            "version": 1,
            "id": self.id,
            "source": "simulation",
            "fps": config.chunk.fps,
            "cameras": list(self._cameras),
            "scene_id": scene.scene_id,
            "scene_path": str(scene.scene_path.resolve()),
            "camera_calibration": [asdict(camera) for camera in scene.cameras],
            "config": asdict(config),
            "frames_path": "frames.jsonl",
            "commands_path": "commands.jsonl",
            "geometry_path": "scene_geometry.json",
            "scene_assets": {},
            "image_encoding": "JPEG quality 95; native model dimensions",
            "status": "recording",
        }
        for name in ("scene.json", "map.json", "tracks.json", "poses.json"):
            source = scene.scene_path / name
            if source.is_file():
                destination = self.path / "scene" / name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(source, destination)
                self._manifest["scene_assets"][name] = f"scene/{name}"
        geometry = {
            name: [asdict(item) for item in getattr(scene, name, ())]
            for name in (
                "line_layers",
                "triangle_layers",
                "polygon_layers",
                "vehicle_bbox_tracks",
            )
        }
        (self.path / "scene_geometry.json").write_text(
            json.dumps(
                geometry,
                default=lambda value: (
                    value.tolist() if isinstance(value, np.ndarray) else str(value)
                ),
                allow_nan=False,
            )
        )
        self._write_manifest()

    def _write_manifest(self) -> None:
        def encode(value: Any) -> Any:
            return value.tolist() if isinstance(value, np.ndarray) else str(value)

        self._manifest["frame_count"] = self.frames
        temporary = self.path / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(self._manifest, default=encode, allow_nan=False, indent=2)
        )
        temporary.replace(self.path / "manifest.json")

    def status(self) -> dict[str, Any]:
        """Return browser-safe recording progress."""
        return {"state": self._manifest["status"], "id": self.id, "frames": self.frames}

    def append(
        self,
        chunk: FrameChunk,
        trajectory: TrajectoryChunk,
        commands: Sequence[DriverCommand],
        command_times_s: Sequence[float],
    ) -> None:
        """Persist generated frames against their matching simulation timestamps."""
        if self._closed:
            raise RuntimeError("Recording is already closed")
        if len(trajectory.vehicle_states) != len(trajectory.timestamps_us):
            raise ValueError(
                "Recording requires one vehicle state per trajectory frame"
            )
        first_us = int(trajectory.timestamps_us[0])
        if self._begin_us is None:
            self._begin_us = first_us
        for i, timestamp in enumerate(trajectory.timestamps_us):
            command_index = (
                bisect_right(command_times_s, (int(timestamp) - first_us) / 1e6) - 1
            )
            self._pending[int(timestamp)] = {
                "rig_to_world": trajectory.rig_poses_world[i].tolist(),
                "vehicle": asdict(trajectory.vehicle_states[i]),
                "command": asdict(commands[command_index]),
                "actors": [
                    {
                        "entity_id": actor.entity_id,
                        "object_type": actor.object_type,
                        "translation_world": actor.translations_world[i].tolist(),
                        "orientation_xyzw": actor.orientations_xyzw[i].tolist(),
                        "dimensions_lwh": actor.dimensions_lwh.tolist(),
                        "is_simulated": actor.is_simulated,
                        "detached_from_track": actor.detached_from_track,
                    }
                    for actor in trajectory.dynamic_actors
                ],
            }
        for command, seconds in zip(commands, command_times_s, strict=True):
            self._commands.write(
                json.dumps(
                    {
                        "timestamp_us": str(first_us + round(seconds * 1e6)),
                        "command": asdict(command),
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        for frame in chunk.frames:
            timestamp = int(frame.timestamp_us)
            if timestamp < self._begin_us:
                continue
            if self._last_us is not None and timestamp <= self._last_us:
                raise ValueError(
                    "Recording timestamps must increase; finalize before resetting"
                )
            sample = self._pending.pop(timestamp)
            views = frame.view_model_rgb_host_uint8 or frame.view_rgb_host_uint8
            if len(views) != len(self._cameras):
                raise ValueError(
                    "Recording camera count differs from the configured rig"
                )
            if self._start_us is None:
                self._start_us = timestamp
            image_paths = {}
            # ponytail: Synchronous JPEG writes preserve every emitted frame;
            # sustained high-rate rigs should batch encoding in a separate process.
            for camera_index, (camera, pixels) in enumerate(
                zip(self._cameras, views, strict=True)
            ):
                name = f"rgb/{camera_index:02d}/{self.frames:06d}.jpg"
                path = self.path / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(pixels)[..., :3]).save(path, quality=95)
                image_paths[camera] = name
            bev = None
            if frame.bev_host_uint8 is not None:
                bev = f"bev/{self.frames:06d}.png"
                (self.path / "bev").mkdir(exist_ok=True)
                Image.fromarray(np.asarray(frame.bev_host_uint8)[..., :3]).save(
                    self.path / bev
                )
            record = {
                "frame_index": self.frames,
                "timestamp_us": str(timestamp),
                "time_s": (timestamp - self._start_us) / 1e6,
                "images": image_paths,
                "bev": bev,
                **sample,
            }
            self._records.write(json.dumps(record, allow_nan=False) + "\n")
            self.frames += 1
            self._last_us = timestamp
        self._records.flush()
        self._commands.flush()

    def close(self, *, error: str | None = None) -> dict[str, Any]:
        """Retain partial data on error and atomically publish completed archives."""
        if self._closed:
            return self.status()
        self._records.close()
        self._commands.close()
        self._closed = True
        self._manifest["status"] = "error" if error or not self.frames else "ready"
        self._manifest["unemitted_frame_count"] = len(self._pending)
        if error:
            self._manifest["error"] = error
        self._write_manifest()
        if self.frames:
            temporary = self.path.with_suffix(".zip.tmp")
            with ZipFile(temporary, "x", compression=ZIP_STORED) as archive:
                for path in sorted(self.path.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(self.path))
            temporary.replace(self.path.with_suffix(".zip"))
        return {**self.status(), **({"message": error} if error else {})}
