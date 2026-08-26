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

"""Offline authoring of the bundled obstacle vehicle-track catalog."""

from __future__ import annotations

import argparse
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from omnidreams_game_engine.ply_io import load_mesh_vf

_OBSTACLE_PATH = "clipgt/obstacle.parquet"
_GROUND_PATH = "mesh_ground.ply"
_GROUND_RADIUS_M = 3.0


def _object_type_for_category(category: str) -> str | None:
    normalized = category.replace("_", " ").replace("-", " ").title().replace(" ", "_")
    if normalized in {
        "Bus",
        "Heavy_Truck",
        "Train_Or_Tram_Car",
        "Trolley_Bus",
        "Trailer",
        "Truck",
    }:
        return "Truck"
    if normalized in {"Vehicle", "Automobile", "Other_Vehicle", "Car"}:
        return "Car"
    return None


class _GroundVertexIndex:
    """Spatial index for median ground-height queries over PLY vertices."""

    def __init__(
        self,
        vertices: npt.NDArray[np.floating],
        *,
        cell_size_m: float = _GROUND_RADIUS_M,
    ) -> None:
        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._cell_size_m = float(cell_size_m)
        cells = np.floor(self._vertices[:, :2] / self._cell_size_m).astype(np.int64)
        keys = self._keys(cells[:, 0], cells[:, 1])
        self._order = np.argsort(keys, kind="stable")
        sorted_keys = keys[self._order]
        unique, starts, counts = np.unique(
            sorted_keys, return_index=True, return_counts=True
        )
        self._ranges = {
            int(key): (int(start), int(start + count))
            for key, start, count in zip(unique, starts, counts, strict=True)
        }

    @staticmethod
    def _keys(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.int64]:
        x_array = np.asarray(x, dtype=np.int64)
        y_array = np.asarray(y, dtype=np.int64)
        return (x_array << np.int64(32)) ^ (y_array & np.int64(0xFFFFFFFF))

    def median_z(
        self, xy: npt.NDArray[np.floating], radius_m: float = _GROUND_RADIUS_M
    ) -> float | None:
        """Return median nearby ground height for one XY position."""
        center_cell = np.floor(np.asarray(xy) / self._cell_size_m).astype(np.int64)
        reach = int(np.ceil(radius_m / self._cell_size_m))
        candidates: list[np.ndarray] = []
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                key = int(self._keys(center_cell[0] + dx, center_cell[1] + dy))
                bounds = self._ranges.get(key)
                if bounds is not None:
                    candidates.append(self._order[bounds[0] : bounds[1]])
        if not candidates:
            return None
        indices = np.concatenate(candidates)
        points = self._vertices[indices]
        near = (
            np.linalg.norm(points[:, :2] - np.asarray(xy)[None, :], axis=1) < radius_m
        )
        if not near.any():
            return None
        return float(np.median(points[near, 2]))


def catalog_arrays_from_records(
    obstacle_rows: Sequence[dict[str, Any]],
    ground_vertices: npt.NDArray[np.floating],
) -> dict[str, np.ndarray]:
    """Convert ClipGT obstacle records into safe concatenated numeric arrays."""
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in obstacle_rows:
        payload = row["obstacle"]
        track_id = str(payload.get("trackline_id", ""))
        if track_id:
            grouped_rows[track_id].append(row)

    ground = _GroundVertexIndex(ground_vertices)
    timestamps_parts: list[np.ndarray] = []
    translations_parts: list[np.ndarray] = []
    orientations_parts: list[np.ndarray] = []
    dimensions: list[np.ndarray] = []
    object_type_codes: list[int] = []
    ground_offsets: list[float] = []
    offsets = [0]

    for track_id in sorted(grouped_rows):
        observations = sorted(
            grouped_rows[track_id], key=lambda row: int(row["key"]["timestamp_micros"])
        )
        object_type = _object_type_for_category(
            str(observations[0]["obstacle"].get("category", "Others"))
        )
        if object_type is None:
            continue
        track_timestamps: list[int] = []
        track_centers: list[list[float]] = []
        track_dimensions: list[list[float]] = []
        track_orientations: list[np.ndarray] = []
        for observation in observations:
            payload = observation["obstacle"]
            center = payload["center"]
            size = payload["size"]
            orientation = payload["orientation"]
            quaternion = np.asarray(
                [
                    float(orientation["x"]),
                    float(orientation["y"]),
                    float(orientation["z"]),
                    float(orientation["w"]),
                ],
                dtype=np.float32,
            )
            if float(np.linalg.norm(quaternion)) <= 1.0e-8:
                continue
            track_timestamps.append(int(observation["key"]["timestamp_micros"]))
            track_centers.append(
                [float(center["x"]), float(center["y"]), float(center["z"])]
            )
            track_dimensions.append(
                [float(size["x"]), float(size["y"]), float(size["z"])]
            )
            track_orientations.append(quaternion)
        if len(track_timestamps) < 2:
            continue

        timestamps = np.asarray(track_timestamps, dtype=np.int64)
        centers = np.asarray(track_centers, dtype=np.float32)
        first_dimensions = np.asarray(track_dimensions[0], dtype=np.float32)
        source_ground_z = ground.median_z(centers[0, :2])
        source_ground_offset = (
            float(first_dimensions[2] * 0.5)
            if source_ground_z is None
            else float(centers[0, 2] - source_ground_z)
        )
        timestamps_parts.append(timestamps - timestamps[0])
        translations_parts.append(centers - centers[0])
        orientations_parts.append(np.asarray(track_orientations, dtype=np.float32))
        dimensions.append(first_dimensions)
        object_type_codes.append(0 if object_type == "Car" else 1)
        ground_offsets.append(source_ground_offset)
        offsets.append(offsets[-1] + len(timestamps))

    if not timestamps_parts:
        raise ValueError("Source scene contains no usable Car or Truck tracks")
    return {
        "format_version": np.asarray(1, dtype=np.int32),
        "sample_offsets": np.asarray(offsets, dtype=np.int64),
        "timestamps_us": np.concatenate(timestamps_parts).astype(np.int64),
        "translations_local_m": np.concatenate(translations_parts).astype(np.float32),
        "orientations_xyzw": np.concatenate(orientations_parts).astype(np.float32),
        "dimensions_lwh": np.asarray(dimensions, dtype=np.float32),
        "object_type_codes": np.asarray(object_type_codes, dtype=np.uint8),
        "source_ground_offsets_m": np.asarray(ground_offsets, dtype=np.float32),
    }


def extract_catalog(source: Path, output: Path, *, force: bool = False) -> None:
    """Extract a deterministic obstacle catalog from an explicit USDZ archive."""
    if output.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output}; pass --force to replace it"
        )
    import pyarrow.parquet as pq

    with zipfile.ZipFile(source) as archive:
        missing = {
            name
            for name in (_OBSTACLE_PATH, _GROUND_PATH)
            if name not in archive.namelist()
        }
        if missing:
            raise FileNotFoundError(
                f"Source scene is missing required entries: {sorted(missing)}"
            )
        with archive.open(_OBSTACLE_PATH) as handle:
            obstacle_rows = pq.read_table(handle).to_pylist()
        ground_vertices, _ = load_mesh_vf(archive.read(_GROUND_PATH))

    arrays = catalog_arrays_from_records(obstacle_rows, ground_vertices)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Crazy Robotaxi obstacle templates from a ClipGT USDZ."
    )
    parser.add_argument("source", type=Path, help="Source ClipGT USDZ archive.")
    parser.add_argument("output", type=Path, help="Destination numeric NPZ catalog.")
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing output file."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the obstacle-template authoring command."""
    args = _parse_args(argv)
    extract_catalog(args.source, args.output, force=args.force)


if __name__ == "__main__":
    main()
