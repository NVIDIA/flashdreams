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

"""Map-independent vehicle-track templates for live-edit obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import IO

import numpy as np
import numpy.typing as npt

_CATALOG_FILENAME = "obstacle_vehicle_tracks_v1.npz"
_CATALOG_FORMAT_VERSION = 1
_OBJECT_TYPES = ("Car", "Truck")


@dataclass(frozen=True)
class ObstacleTemplate:
    """One vehicle trajectory normalized to its first center and timestamp."""

    template_index: int
    """Stable zero-based index in the bundled catalog."""

    object_type: str
    """Renderer object type."""

    timestamps_us: npt.NDArray[np.int64]
    """Sample timestamps relative to the first sample."""

    translations_local_m: npt.NDArray[np.float32]
    """Sample centers relative to the first sample."""

    orientations_xyzw: npt.NDArray[np.float32]
    """Source orientation quaternion for every sample."""

    dimensions_lwh: npt.NDArray[np.float32]
    """Box dimensions from the first source sample."""

    source_ground_offset_m: float
    """First center height above the source scene's local ground."""

    @property
    def drift_m(self) -> float:
        """Return ground-plane displacement between the endpoint samples."""
        return float(np.linalg.norm(self.translations_local_m[-1, :2]))

    @property
    def duration_s(self) -> float:
        """Return track coverage in seconds."""
        return float(self.timestamps_us[-1]) * 1.0e-6

    @property
    def motion_heading_rad(self) -> float:
        """Return the endpoint ground-plane motion heading."""
        motion = self.translations_local_m[-1, :2]
        return float(np.arctan2(motion[1], motion[0]))

    @property
    def sampled_speed_mps(self) -> float:
        """Return average speed along the sampled ground-plane path."""
        elapsed_s = np.diff(self.timestamps_us).astype(np.float64) * 1.0e-6
        distances_m = np.linalg.norm(
            np.diff(self.translations_local_m[:, :2], axis=0), axis=1
        )
        valid = elapsed_s > 0.0
        if not valid.any():
            return 0.0
        return float(np.sum(distances_m[valid]) / np.sum(elapsed_s[valid]))


@dataclass(frozen=True)
class ObstacleTemplateCatalog:
    """Validated collection of obstacle vehicle trajectories."""

    templates: tuple[ObstacleTemplate, ...]
    """Vehicle tracks in deterministic source-track order."""

    def moving(
        self,
        *,
        min_drift_m: float,
        min_coverage_s: float,
        length_range_m: tuple[float, float],
    ) -> tuple[ObstacleTemplate, ...]:
        """Return PR494-compatible moving templates in selection order."""
        lo, hi = length_range_m
        selected = [
            template
            for template in self.templates
            if len(template.timestamps_us) >= 8
            and template.duration_s >= min_coverage_s
            and lo <= float(template.dimensions_lwh.max()) <= hi
            and template.drift_m >= min_drift_m
        ]

        def order_key(template: ObstacleTemplate) -> tuple[int, float]:
            speed_mps = template.drift_m / template.duration_s
            return (0 if 2.0 <= speed_mps <= 8.0 else 1, -template.duration_s)

        selected.sort(key=order_key)
        return tuple(selected)

    def parked(
        self, *, length_range_m: tuple[float, float]
    ) -> tuple[ObstacleTemplate, ...]:
        """Return PR494-compatible parked templates in selection order."""
        lo, hi = length_range_m
        selected = [
            template
            for template in self.templates
            if len(template.timestamps_us) >= 8
            and template.duration_s >= 3.0
            and lo <= float(template.dimensions_lwh.max()) <= hi
            and template.drift_m < 2.0
        ]
        selected.sort(key=lambda template: -template.duration_s)
        return tuple(selected)


def _require_array(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    dtype: npt.DTypeLike,
    ndim: int,
) -> np.ndarray:
    if name not in archive.files:
        raise ValueError(f"Obstacle template catalog is missing {name!r}")
    value = np.asarray(archive[name])
    if value.dtype != np.dtype(dtype) or value.ndim != ndim:
        raise ValueError(
            f"Obstacle template catalog {name!r} must have dtype {np.dtype(dtype)} "
            f"and {ndim} dimensions, got {value.dtype} and {value.ndim}"
        )
    return value


def load_obstacle_template_catalog_from_file(
    source: str | Path | IO[bytes],
) -> ObstacleTemplateCatalog:
    """Load and validate a safe numeric obstacle-template archive."""
    with np.load(source, allow_pickle=False) as archive:
        version = _require_array(archive, "format_version", dtype=np.int32, ndim=0)
        if int(version) != _CATALOG_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported obstacle template catalog version {int(version)}"
            )
        offsets = _require_array(archive, "sample_offsets", dtype=np.int64, ndim=1)
        timestamps = _require_array(archive, "timestamps_us", dtype=np.int64, ndim=1)
        translations = _require_array(
            archive, "translations_local_m", dtype=np.float32, ndim=2
        )
        orientations = _require_array(
            archive, "orientations_xyzw", dtype=np.float32, ndim=2
        )
        dimensions = _require_array(archive, "dimensions_lwh", dtype=np.float32, ndim=2)
        object_type_codes = _require_array(
            archive, "object_type_codes", dtype=np.uint8, ndim=1
        )
        ground_offsets = _require_array(
            archive, "source_ground_offsets_m", dtype=np.float32, ndim=1
        )

        arrays = tuple(
            np.array(value, copy=True)
            for value in (
                offsets,
                timestamps,
                translations,
                orientations,
                dimensions,
                object_type_codes,
                ground_offsets,
            )
        )
    (
        offsets,
        timestamps,
        translations,
        orientations,
        dimensions,
        object_type_codes,
        ground_offsets,
    ) = arrays
    _validate_catalog_arrays(
        offsets=offsets,
        timestamps=timestamps,
        translations=translations,
        orientations=orientations,
        dimensions=dimensions,
        object_type_codes=object_type_codes,
        ground_offsets=ground_offsets,
    )

    templates = []
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        templates.append(
            ObstacleTemplate(
                template_index=index,
                object_type=_OBJECT_TYPES[int(object_type_codes[index])],
                timestamps_us=timestamps[int(start) : int(end)],
                translations_local_m=translations[int(start) : int(end)],
                orientations_xyzw=orientations[int(start) : int(end)],
                dimensions_lwh=dimensions[index],
                source_ground_offset_m=float(ground_offsets[index]),
            )
        )
    return ObstacleTemplateCatalog(templates=tuple(templates))


def _validate_catalog_arrays(
    *,
    offsets: np.ndarray,
    timestamps: np.ndarray,
    translations: np.ndarray,
    orientations: np.ndarray,
    dimensions: np.ndarray,
    object_type_codes: np.ndarray,
    ground_offsets: np.ndarray,
) -> None:
    track_count = len(offsets) - 1
    sample_count = len(timestamps)
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != sample_count:
        raise ValueError("Obstacle template sample offsets are inconsistent")
    if np.any(np.diff(offsets) < 2):
        raise ValueError("Every obstacle template must have at least two samples")
    if translations.shape != (sample_count, 3):
        raise ValueError("Obstacle template translations must have shape [samples, 3]")
    if orientations.shape != (sample_count, 4):
        raise ValueError("Obstacle template orientations must have shape [samples, 4]")
    if dimensions.shape != (track_count, 3):
        raise ValueError("Obstacle template dimensions must have shape [tracks, 3]")
    if object_type_codes.shape != (track_count,) or np.any(
        object_type_codes >= len(_OBJECT_TYPES)
    ):
        raise ValueError("Obstacle template object type codes are invalid")
    if ground_offsets.shape != (track_count,):
        raise ValueError("Obstacle template ground offsets must have shape [tracks]")
    numeric = (translations, orientations, dimensions, ground_offsets)
    if any(not np.isfinite(value).all() for value in numeric):
        raise ValueError("Obstacle template catalog contains non-finite values")
    if np.any(dimensions <= 0.0):
        raise ValueError("Obstacle template dimensions must be positive")
    if np.any(np.linalg.norm(orientations, axis=1) <= 1.0e-8):
        raise ValueError("Obstacle template orientations must be non-zero")
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        track_timestamps = timestamps[int(start) : int(end)]
        if track_timestamps[0] != 0 or np.any(np.diff(track_timestamps) <= 0):
            raise ValueError(
                "Obstacle template timestamps must start at zero and increase"
            )
        if not np.allclose(translations[int(start)], 0.0, atol=1.0e-5):
            raise ValueError("Obstacle template translations must start at zero")


@lru_cache(maxsize=1)
def load_obstacle_template_catalog() -> ObstacleTemplateCatalog:
    """Load the obstacle catalog bundled with Crazy Robotaxi."""
    resource = files("crazy_robotaxi.assets").joinpath(_CATALOG_FILENAME)
    with resource.open("rb") as handle:
        return load_obstacle_template_catalog_from_file(handle)


__all__ = [
    "ObstacleTemplate",
    "ObstacleTemplateCatalog",
    "load_obstacle_template_catalog",
    "load_obstacle_template_catalog_from_file",
]
