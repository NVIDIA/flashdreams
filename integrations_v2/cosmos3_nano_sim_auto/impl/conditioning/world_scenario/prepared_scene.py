# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned plain-text Ludus scenes and their synchronized opening images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ludus_renderer import (
    CUBE_FLAG_FLAT_FACES,
    CUBE_FLAG_WIREFRAME,
    PRIM_OBSTACLE,
    PRIM_TRAFFIC_LIGHT,
    PRIM_TRAFFIC_SIGN,
    CubePool,
    TimestampedPolygonPool,
    TimestampedPolylinePool,
    TimestampedScene,
)

from cosmos3_nano_sim_auto.impl.conditioning.camera_rig import CAMERAS

from . import camera as camera_math

if TYPE_CHECKING:
    from .prepared_lidar import PreparedLidar

FORMAT = "flashdreams.ludus-scene"
VERSION = 1
MAX_JSON_BYTES = 512 * 1024 * 1024
"""Per-file ceiling before JSON parsing; large scenes must use another format version."""

RENDER_SETTINGS = {
    "max_extrapolation_us": 0,
    "width_in_ndc": True,
    "msaa_samples": 4,
    "depth_fade": False,
    "cull_behind_camera": True,
    "outline_width": camera_math.OUTLINE_WIDTH,
    "camera_depth_max": 200.0,
}
"""Renderer settings expected by the prepared scene format."""


def configure(context: Any) -> None:
    """Apply the common renderer settings before uploading a scene."""
    for name, value in RENDER_SETTINGS.items():
        if name not in ("outline_width", "camera_depth_max"):
            getattr(context, f"set_{name}")(value)


def sha256(path: Path) -> str:
    """Hash an asset without reading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset(root: Path, name: str) -> Path:
    """Resolve a relative asset inside the bundle, refusing escaping paths."""
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset must be inside the scene bundle: {name}")
    return path


def read_json(path: Path) -> Any:
    """Read bounded UTF-8 JSON, rejecting nonfinite numbers."""
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Scene JSON exceeds {MAX_JSON_BYTES} bytes: {path}")

    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def write_json(path: Path, value: Any) -> None:
    """Write readable, deterministic JSON with compact numeric rows."""

    def encode(item, depth=0):
        if not isinstance(item, (dict, list)) or not item:
            return json.dumps(item, allow_nan=False)
        if isinstance(item, list) and all(
            not isinstance(row, (list, dict)) for row in item
        ):
            return json.dumps(item, allow_nan=False, separators=(",", ":"))
        indent = "  " * (depth + 1)
        if isinstance(item, dict):
            opening, closing = "{", "}"
            rows = (
                json.dumps(str(key)) + ": " + encode(row, depth + 1)
                for key, row in item.items()
            )
        else:
            opening, closing = "[", "]"
            rows = (encode(row, depth + 1) for row in item)
        return (
            opening
            + "\n"
            + ",\n".join(indent + row for row in rows)
            + "\n"
            + "  " * depth
            + closing
        )

    path.write_text(encode(value) + "\n", encoding="utf-8")


def _array(value, shape, dtype=np.float32):
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or result.shape != tuple(shape):
        raise ValueError(f"Expected numeric array of shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Scene arrays must be finite")
    if np.issubdtype(dtype, np.integer):
        if result.dtype.kind not in "iu" or np.any(result < 0):
            raise ValueError("Indices must be nonnegative integers")
        if np.any(result > np.iinfo(dtype).max):
            raise ValueError("Index exceeds supported range")
    result = result.astype(dtype)
    if not np.isfinite(result).all():
        raise ValueError("Scene array exceeds its numeric type")
    return result


def _stamps(values):
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(v, str) or not v.isdecimal() for v in values)
    ):
        raise ValueError("Timestamps must be nonnegative decimal strings")
    integers = [int(v) for v in values]
    if any(v > np.iinfo(np.int64).max for v in integers):
        raise ValueError("Timestamp exceeds int64 range")
    result = np.array(integers, dtype=np.int64)
    if np.any(result[1:] <= result[:-1]):
        raise ValueError("Track timestamps must be strictly increasing")
    return result


def map_data(scene: TimestampedScene) -> dict:
    """Export static Ludus segments and triangles in pool order.

    Version 1 accepts the static two-point lines and triangle soups produced
    by the prepared scene format. Refuse other pool layouts instead of flattening
    their time or topology.
    """
    result = {}
    for name, pools, corners in (
        ("lines", scene.polyline_pools, 2),
        ("triangles", scene.polygon_pools, 3),
    ):
        result[name] = []
        for pool in pools:
            count = len(pool.varrays_prefix_sum)
            if (
                pool.timestamps_us.tolist() != [0]
                or pool.timestamped_varrays_prefix_sum.tolist() != [count]
                or pool.varrays_prefix_sum.tolist()
                != list(range(corners, count * corners + 1, corners))
                or (
                    isinstance(pool, TimestampedPolygonPool)
                    and (
                        pool.triangle_prefix_sum.tolist() != list(range(1, count + 1))
                        or pool.triangles.tolist() != [[0, 1, 2]] * count
                    )
                )
            ):
                raise ValueError(
                    "Version 1 requires static segments and triangle soups"
                )
            result[name].append(
                {
                    "prim": pool.prim_type_id,
                    "vertices": pool.vertices.cpu().reshape(count, corners, 3).tolist(),
                }
            )
    return result


def track_data(scene: TimestampedScene) -> list:
    """Export ordered cuboid tracks with exact lifetimes and face colors."""
    result = []
    for pool in scene.cube_pools or ():
        tracks = []
        first = 0
        for index, last in enumerate(pool.cube_ts_prefix_sum.tolist()):
            tracks.append(
                {
                    "timestamps_us": [
                        str(v) for v in pool.track_timestamps_us[first:last].tolist()
                    ],
                    "translations": pool.translations[first:last].tolist(),
                    "quaternions_xyzw": pool.quaternions[first:last].tolist(),
                    "scale": pool.scales[index].tolist(),
                    "colors": pool.colors[index].tolist(),
                }
            )
            first = last
        if (
            sorted({int(t) for track in tracks for t in track["timestamps_us"]})
            != pool.timestamps_us.tolist()
        ):
            raise ValueError(
                "Cube observation times must equal the union of its tracks"
            )
        result.append(
            {"prim": pool.prim_type_id, "flags": pool.render_flags, "tracks": tracks}
        )
    return result


def reconstruct(
    maps: dict, tracks: list, palette: dict, widths: dict
) -> TimestampedScene:
    """Validate and reconstruct CPU pools, deriving all GPU offsets from geometry."""
    if set(maps) != {"lines", "triangles"}:
        raise ValueError("Unsupported map pool categories")
    lines, polygons, cubes = [], [], []
    for name, corners, target in (("lines", 2, lines), ("triangles", 3, polygons)):
        for group in maps[name]:
            prim = group["prim"]
            if (
                type(prim) is not int
                or prim not in palette
                or (corners == 2 and prim not in widths)
            ):
                raise ValueError("Map primitive has no palette or line width entry")
            count = len(group["vertices"])
            if not 0 < count < 2**31 // corners:
                raise ValueError("Invalid map pool size")
            vertices = _array(group["vertices"], (count, corners, 3))
            kwargs = dict(
                timestamps_us=torch.tensor([0], dtype=torch.int64),
                timestamped_varrays_prefix_sum=torch.tensor([count], dtype=torch.int32),
                varrays_prefix_sum=torch.arange(
                    corners, count * corners + 1, corners, dtype=torch.int32
                ),
                vertices=torch.from_numpy(vertices.reshape(-1, 3)),
            )
            if corners == 2:
                target.append(TimestampedPolylinePool(**kwargs, prim_type_id=prim))
            else:
                target.append(
                    TimestampedPolygonPool(
                        **kwargs,
                        prim_type_id=prim,
                        triangle_prefix_sum=torch.arange(
                            1, count + 1, dtype=torch.int32
                        ),
                        triangles=torch.tensor([[0, 1, 2]] * count, dtype=torch.int32),
                    )
                )
    for group in tracks:
        if type(group["prim"]) is not int or group["prim"] not in (
            PRIM_OBSTACLE,
            PRIM_TRAFFIC_LIGHT,
            PRIM_TRAFFIC_SIGN,
        ):
            raise ValueError("Unsupported cube primitive")
        if type(group["flags"]) is not int or group["flags"] & ~(
            CUBE_FLAG_FLAT_FACES | CUBE_FLAG_WIREFRAME
        ):
            raise ValueError("Unsupported cube render flags")
        lengths, times, translations, quaternions, scales, colors = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for track in group["tracks"]:
            stamps = _stamps(track["timestamps_us"])
            count = len(stamps)
            translation = _array(track["translations"], (count, 3))
            quaternion = _array(track["quaternions_xyzw"], (count, 4))
            scale = _array(track["scale"], (3,))
            color = _array(track["colors"], (6,))
            if not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-5):
                raise ValueError("Invalid cube quaternion")
            if np.any(scale <= 0) or np.any((color < 0) | (color > 1)):
                raise ValueError("Invalid cube extent or color")
            lengths.append(count)
            times.append(stamps)
            translations.append(translation)
            quaternions.append(quaternion)
            scales.append(scale)
            colors.append(color)
        if not lengths or sum(lengths) >= 2**31:
            raise ValueError("Invalid cube pool size")
        stamps = np.concatenate(times)
        cubes.append(
            CubePool(
                timestamps_us=torch.from_numpy(np.unique(stamps)),
                cube_ts_prefix_sum=torch.from_numpy(
                    np.cumsum(lengths).astype(np.int32)
                ),
                track_timestamps_us=torch.from_numpy(stamps),
                translations=torch.from_numpy(np.concatenate(translations)),
                quaternions=torch.from_numpy(np.concatenate(quaternions)),
                scales=torch.from_numpy(np.stack(scales)),
                colors=torch.from_numpy(np.stack(colors)),
                prim_type_id=group["prim"],
                render_flags=group["flags"],
            )
        )
    return TimestampedScene(lines, polygons, cubes)


@dataclass
class PreparedScene:
    """Validated CPU scene retained across uploads and interactive restarts."""

    root: Path
    """Bundle directory."""
    manifest: dict
    """Scene identity, camera calibration, timing, styles, and asset hashes."""
    scene: TimestampedScene
    """CPU geometry pools; parsing happens once when opening the scene."""
    tracks: dict[str, np.ndarray]
    """Float32 camera-to-world pose sequences in OpenCV RDF coordinates."""
    palette: dict[int, tuple[int, ...]]
    """RGBA bytes indexed by primitive type."""
    widths: dict[int, float]
    """Native-pixel line widths indexed by primitive type."""
    lidar: PreparedLidar | None = None
    """Optional validated v2 LiDAR calibration and semantic geometry."""

    def upload_data(self, device) -> TimestampedScene:
        """Transfer validated pools to the renderer's device."""

        def move(pool):
            return type(pool)(
                **{
                    field.name: value.to(device)
                    if isinstance(value := getattr(pool, field.name), torch.Tensor)
                    else value
                    for field in fields(pool)
                }
            )

        return TimestampedScene(
            [move(p) for p in self.scene.polyline_pools],
            [move(p) for p in self.scene.polygon_pools],
            [move(p) for p in self.scene.cube_pools or ()],
        )

    def intrinsics(self, name: str) -> np.ndarray:
        """Return the original calibration for a named camera."""
        return np.asarray(
            self.manifest["cameras"][name]["intrinsics"], dtype=np.float64
        )

    def frame_mapping(self, start_frame: int, wanted: int | None) -> np.ndarray:
        """Map the opening-image sample's frame zero onto recorded poses."""
        if start_frame != 0:
            raise ValueError(
                "Prepared RGB supplies only sample frame 0; export another opening state"
            )
        first = self.manifest["opening_frame"]
        count = self.manifest["frames"] - first
        if wanted is not None and not 0 < wanted <= count:
            raise ValueError(
                f"Prepared scene has {count} frames after its opening state"
            )
        return np.arange(first, first + (count if wanted is None else wanted))


def load(root: Path) -> PreparedScene:
    """Load geometry, calibration, prompts and opening images from a scene bundle."""
    root = root.resolve()
    manifest = read_json(root / "scene.json")
    if (
        type(manifest["version"]) is not int
        or manifest["format"] != FORMAT
        or manifest["version"] not in (VERSION, 2)
    ):
        raise ValueError("Unsupported prepared scene format/version")
    if manifest["render_settings"] != RENDER_SETTINGS:
        raise ValueError("Unsupported prepared scene rendering settings")
    if (
        manifest["camera_order"] != list(CAMERAS[:7])
        or manifest["preparation_anchor"] != CAMERAS[0]
    ):
        raise ValueError(
            "Expected the canonical seven-camera rig and front-wide preparation anchor"
        )
    if manifest["coordinates"] != "metres; OpenCV RDF camera-to-world":
        raise ValueError("Unsupported scene coordinates")
    count, opening = manifest["frames"], manifest["opening_frame"]
    if (
        type(count) is not int
        or not 0 < count <= 1_000_000
        or type(opening) is not int
        or not 0 <= opening < count
    ):
        raise ValueError("Invalid scene frame count or opening frame")
    start, end = _stamps([manifest["start_us"], manifest["end_us"]])
    fps = camera_math.frame_rate(count, int(start), int(end))
    expected_files = {"map.json", "tracks.json", "poses.json", "sample.json"}
    if manifest["version"] == 2:
        if not np.isclose(fps, 30, rtol=0, atol=1e-3):
            raise ValueError("Prepared LiDAR requires a 30 FPS scene clock")
        expected_files.add("lidar.json")
    expected_files.update(
        f"{directory}/{name}.png"
        for directory in ("initial_rgb", "controls")
        for name in CAMERAS[:7]
    )
    if set(manifest["files_sha256"]) != expected_files:
        raise ValueError("Incomplete prepared scene asset inventory")
    for name, digest in manifest["files_sha256"].items():
        path = asset(root, name)
        if sha256(path) != digest:
            raise ValueError(f"Scene asset hash mismatch: {name}")
        if path.suffix == ".png":
            from PIL import Image

            with Image.open(path) as image:
                if (
                    image.format != "PNG"
                    or image.mode != "RGB"
                    or image.size != (1280, 720)
                ):
                    raise ValueError(f"Expected a 1280x720 RGB PNG: {name}")
                image.verify()
    palette = {}
    for key, value in manifest["palette"].items():
        if not key.isdecimal() or not 0 <= int(key) < 65536:
            raise ValueError("Invalid palette primitive ID")
        color = _array(value, (4,), np.int32)
        if np.any(color > 255):
            raise ValueError("Palette channels must be bytes")
        palette[int(key)] = tuple(int(v) for v in color)
    widths = {int(k): float(v) for k, v in manifest["widths"].items()}
    if any(k not in palette or not np.isfinite(v) or v <= 0 for k, v in widths.items()):
        raise ValueError("Invalid primitive width")
    poses = read_json(root / "poses.json")
    tracks = {}
    if set(poses) != set(CAMERAS[:7]) or set(manifest["cameras"]) != set(poses):
        raise ValueError("Missing or unexpected camera pose tracks")
    for name in CAMERAS[:7]:
        track = _array(poses[name], (count, 4, 4))
        rotation = track[:, :3, :3]
        if (
            not np.allclose(track[:, 3], [0, 0, 0, 1])
            or not np.allclose(
                rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5
            )
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5)
        ):
            raise ValueError(f"Invalid rigid camera poses: {name}")
        intrinsic = _array(manifest["cameras"][name]["intrinsics"], (14,), np.float64)
        if np.any(intrinsic[2:4] <= 0):
            raise ValueError(f"Invalid camera dimensions: {name}")
        tracks[name] = track
    sample = read_json(root / "sample.json")
    expected_start = {
        "clip_id": manifest["clip_id"],
        "frame_index": opening,
        "timestamp_us": str(int(start) + round(opening * 1e6 / fps)),
    }
    if (
        sample.get("scene_start") != expected_start
        or sample.get("prepared_scene") != "scene.json"
    ):
        raise ValueError("Opening sample and scene start disagree")
    if (
        sample.get("model_mode") != "image2video"
        or not sample.get("prompt", "").strip()
    ):
        raise ValueError("Prepared scene needs opening RGB and a prompt")
    span = sample.get("frame_range")
    if sample.get("resolution") != "480" or (
        span is not None
        and (
            not isinstance(span, list)
            or len(span) != 2
            or span[0] != 0
            or not isinstance(span[1], int)
            or span[1] <= 0
        )
    ):
        raise ValueError(
            "Prepared sample must use 480 resolution and its recorded opening"
        )
    expected_views = [
        {
            "camera_key": name,
            "image_path": f"initial_rgb/{name}.png",
            "control_path": f"controls/{name}.png",
        }
        for name in CAMERAS[:7]
    ]
    views = sample["multiview"]["views"]
    captions = [view.get("caption") for view in views if "caption" in view]
    if (captions and len(captions) != len(expected_views)) or [
        {key: value for key, value in view.items() if key != "caption"}
        for view in views
    ] != expected_views:
        raise ValueError("Opening sample has wrong camera order or asset paths")
    scene = reconstruct(
        read_json(root / "map.json"), read_json(root / "tracks.json"), palette, widths
    )
    lidar = None
    if manifest["version"] == 2:
        from . import prepared_lidar

        lidar = prepared_lidar.load(read_json(asset(root, "lidar.json")), manifest)
    return PreparedScene(root, manifest, scene, tracks, palette, widths, lidar)
