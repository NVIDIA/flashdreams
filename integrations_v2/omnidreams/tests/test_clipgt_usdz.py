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

"""Tests for checkpoint-free ClipGT scene bundle generation."""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from interactive_drive.config import RasterConfig
from interactive_drive.ply_io import load_mesh_vf
from interactive_drive.scene_loader import load_scene_bundle
from omnidreams.impl.tools.clipgt_usdz import (
    _ground_ribbon_mesh,
    build_alpasim_clipgt_usdz,
)
from PIL import Image

pytestmark = pytest.mark.ci_cpu

_CLIP_ID = "00000000-0000-0000-0000-000000000001_1000000_2000000"
_CAMERA = "camera:front:wide:120fov"
_CAMERA_LOGICAL_NAME = "camera_front_wide_120fov"
_REQUIRED_MAP_FILES = (
    "lane.parquet",
    "road_boundary.parquet",
    "traffic_sign.parquet",
    "wait_line.parquet",
)


def _key(timestamp_us: int, label: str, *, clip_id: str = _CLIP_ID) -> dict:
    return {
        "clip_id": clip_id,
        "timestamp_micros": timestamp_us,
        "label_class_id": label,
    }


def _map_key(label: str, map_id: str) -> dict:
    return {
        "clip_id": _CLIP_ID,
        "label_class_id": label,
        "map_id": map_id,
        "map_id_version": "v1",
    }


def _expected_scene_id(clip_id: str = _CLIP_ID) -> str:
    scene_uuid = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"flashdreams:clipgt:{clip_id}:camera:{_CAMERA_LOGICAL_NAME}",
    )
    return f"clipgt-{scene_uuid}"


def _write_parquet(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def _calibration_rows() -> list[dict]:
    rig = {
        "rig": {
            "sensors": [
                {
                    "name": _CAMERA,
                    "protocol": "camera.virtual",
                    "nominalSensor2Rig_FLU": {
                        "roll-pitch-yaw": [0.0, 0.0, 0.0],
                        "t": [1.7, 0.0, 1.45],
                    },
                    "properties": {
                        "Model": "ftheta",
                        "width": "1920",
                        "height": "1080",
                        "cx": "960",
                        "cy": "540",
                        "polynomial": "0 0.001 0 0 0 0",
                        "polynomial-type": "pixeldistance-to-angle",
                        "linear-c": "1",
                        "linear-d": "0",
                        "linear-e": "0",
                    },
                }
            ]
        }
    }
    return [
        {
            "key": _key(1_000_000, "calibration_estimate"),
            "calibration_estimate": {
                "name": "LIDAR_CAMERA_ALIGNED",
                "rig_json": json.dumps(rig),
            },
            "version": 1,
        }
    ]


def _egomotion_rows(*, clip_id: str = _CLIP_ID) -> list[dict]:
    rows = []
    for index, timestamp_us in enumerate((1_000_000, 1_033_333, 1_066_666)):
        rows.append(
            {
                "key": _key(timestamp_us, "egomotion", clip_id=clip_id),
                "egomotion_estimate": {
                    "name": "egomotion",
                    "location": {"x": float(index), "y": 0.0, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "version": 1,
            }
        )
    return rows


def _lane_line_rows() -> list[dict]:
    return [
        {
            "key": {
                "clip_id": _CLIP_ID,
                "label_class_id": "lane_line",
                "map_id": "map-1",
                "map_id_version": "v1",
            },
            "lane_line": {
                "line_rail": [
                    {"x": 0.0, "y": -2.0, "z": 0.0},
                    {"x": 10.0, "y": -2.0, "z": 0.0},
                ],
                "styles": ["SOLID_SINGLE", "SOLID_SINGLE"],
                "colors": ["WHITE", "WHITE"],
                "left_driving_direction": ["FORWARD", "FORWARD"],
                "right_driving_direction": ["FORWARD", "FORWARD"],
                "egomotion_label_class_id": "",
            },
            "version": 1,
        }
    ]


def _road_boundary_rows() -> list[dict]:
    return [
        {
            "key": _map_key("road_boundary", "boundary-1"),
            "road_boundary": {
                "category": "CURB",
                "location": [
                    {"x": 0.0, "y": 4.0, "z": 0.0},
                    {"x": 10.0, "y": 4.0, "z": 0.0},
                ],
                "left_driving_direction": ["FORWARD", "FORWARD"],
                "right_driving_direction": ["FORWARD", "FORWARD"],
                "is_first_point_physical": True,
                "is_last_point_physical": True,
                "egomotion_label_class_id": "",
            },
            "version": 1,
        }
    ]


def _traffic_sign_rows() -> list[dict]:
    return [
        {
            "key": _map_key("traffic_sign", "sign-1"),
            "traffic_sign": {
                "center": {"x": 8.0, "y": 4.0, "z": 2.0},
                "dimensions": {"x": 0.6, "y": 0.1, "z": 0.6},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "category": "STOP_SIGN",
                "egomotion_label_class_id": "",
            },
            "version": 1,
        }
    ]


def _wait_line_rows() -> list[dict]:
    return [
        {
            "key": _map_key("wait_line", "wait-line-1"),
            "wait_line": {
                "category": "STOP_LINE",
                "location": [
                    {"x": 7.0, "y": -2.0, "z": 0.0},
                    {"x": 7.0, "y": 2.0, "z": 0.0},
                ],
                "is_implicit": False,
                "intersection_subtype": "NONE",
                "egomotion_label_class_id": "",
            },
            "version": 1,
        }
    ]


def _lane_rows() -> list[dict]:
    return [
        {
            "key": {
                "clip_id": _CLIP_ID,
                "label_class_id": "lane",
                "map_id": "lane-1",
                "map_id_version": "v1",
            },
            "lane": {
                "left_rail": [
                    {"x": 0.0, "y": 2.0, "z": 0.0},
                    {"x": 10.0, "y": 2.0, "z": 0.0},
                ],
                "right_rail": [
                    {"x": 0.0, "y": -2.0, "z": 0.0},
                    {"x": 10.0, "y": -2.0, "z": 0.0},
                ],
                "left_edge_styles": ["SOLID_SINGLE", "SOLID_SINGLE"],
                "right_edge_styles": ["SOLID_SINGLE", "SOLID_SINGLE"],
                "egomotion_label_class_id": "",
            },
            "version": 1,
        }
    ]


def _obstacle_rows() -> list[dict]:
    rows = []
    for index, timestamp_us in enumerate((1_000_000, 1_033_333, 1_066_666)):
        rows.append(
            {
                "key": {
                    **_key(timestamp_us, "object_fused"),
                    "label_id": "car-1",
                },
                "obstacle": {
                    "trackline_id": "car-1",
                    "center": {"x": 8.0 + index, "y": 0.0, "z": 0.8},
                    "size": {"x": 4.5, "y": 2.0, "z": 1.6},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    "category": "Automobile",
                    "egomotion_label_class_id": "",
                },
                "version": 1,
            }
        )
    return rows


def _make_clipgt_dir(tmp_path: Path, *, ego_clip_id: str = _CLIP_ID) -> Path:
    clipgt = tmp_path / "clipgt"
    clipgt.mkdir()
    _write_parquet(clipgt / "calibration_estimate.parquet", _calibration_rows())
    _write_parquet(
        clipgt / "egomotion_estimate.parquet",
        _egomotion_rows(clip_id=ego_clip_id),
    )
    _write_parquet(clipgt / "lane.parquet", _lane_rows())
    _write_parquet(clipgt / "lane_line.parquet", _lane_line_rows())
    _write_parquet(clipgt / "road_boundary.parquet", _road_boundary_rows())
    _write_parquet(clipgt / "traffic_sign.parquet", _traffic_sign_rows())
    _write_parquet(clipgt / "wait_line.parquet", _wait_line_rows())
    _write_parquet(clipgt / "obstacle.parquet", _obstacle_rows())
    return clipgt


def _make_jpeg(tmp_path: Path) -> Path:
    path = tmp_path / "first.jpeg"
    Image.new("RGB", (320, 180), color=(20, 80, 140)).save(path, format="JPEG")
    return path


def test_build_alpasim_clipgt_usdz_round_trip(tmp_path: Path) -> None:
    clipgt = _make_clipgt_dir(tmp_path)
    first_frame = _make_jpeg(tmp_path)
    output = build_alpasim_clipgt_usdz(
        tmp_path / "scene.usdz",
        clipgt_dir=clipgt,
        initial_frame_path=first_frame,
        training_date="2026-07-23",
    )

    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert {
            "metadata.yaml",
            "rig_trajectories.json",
            "sequence_tracks.json",
            "mesh_ground.ply",
            "first_image.png",
            "frames/camera_front_wide_120fov/1000000.jpeg",
            "clipgt/calibration_estimate.parquet",
            "clipgt/egomotion_estimate.parquet",
            "clipgt/clip.parquet",
            "clipgt/association.parquet",
            "clipgt/lane.parquet",
            "clipgt/lane_line.parquet",
            "clipgt/road_boundary.parquet",
            "clipgt/traffic_sign.parquet",
            "clipgt/wait_line.parquet",
            "clipgt/obstacle.parquet",
        } <= set(archive.namelist())

        metadata = yaml.safe_load(archive.read("metadata.yaml"))
        assert metadata["scene_id"] == _expected_scene_id()
        assert metadata["source_clip_id"] == _CLIP_ID
        assert metadata["time_range"] == {"start": 1_000_000, "end": 1_066_666}

        rig = json.loads(archive.read("rig_trajectories.json"))
        trajectory = rig["rig_trajectories"][0]
        assert trajectory["sequence_id"] == metadata["scene_id"]
        assert len(trajectory["T_rig_worlds"]) == 3
        assert trajectory["cameras_frame_timestamps_us"]["camera_front_wide_120fov"][
            0
        ] == [966_667, 1_000_000]
        assert trajectory["rig_bbox"]["dim"] == [5.393, 2.109, 1.503]

        tracks = json.loads(archive.read("sequence_tracks.json"))[metadata["scene_id"]]
        assert tracks["tracks_data"]["tracks_id"] == ["car-1"]
        assert tracks["tracks_data"]["tracks_flags"] == [["CONTROLLABLE"]]
        assert len(tracks["tracks_data"]["tracks_poses"][0]) == 3

        jpeg = archive.read("frames/camera_front_wide_120fov/1000000.jpeg")
        assert jpeg.startswith(b"\xff\xd8\xff")
        vertices, faces = load_mesh_vf(archive.read("mesh_ground.ply"))
        assert vertices.shape == (10, 3)
        assert faces.shape == (8, 3)

    bundle = load_scene_bundle(
        scene_path=output,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=320, height=176),
    )
    assert bundle.scene_id == _expected_scene_id()
    assert bundle.initial_rgb.shape == (176, 320, 3)
    assert len(bundle.vehicle_bbox_tracks) == 1
    assert bundle.ground_mesh_faces is not None
    assert bundle.ground_mesh_faces.shape == (8, 3)


def test_build_rejects_mixed_clip_ids(tmp_path: Path) -> None:
    clipgt = _make_clipgt_dir(tmp_path, ego_clip_id="different-clip")
    with pytest.raises(ValueError, match="one identical key.clip_id"):
        build_alpasim_clipgt_usdz(
            tmp_path / "scene.usdz",
            clipgt_dir=clipgt,
            initial_frame_path=_make_jpeg(tmp_path),
        )


def test_build_finds_camera_in_later_calibration_row(tmp_path: Path) -> None:
    clipgt = _make_clipgt_dir(tmp_path)
    calibration_rows = _calibration_rows()
    calibration_rows.insert(
        0,
        {
            "key": _key(900_000, "calibration_estimate"),
            "calibration_estimate": {
                "name": "UNRELATED_RIG",
                "rig_json": json.dumps({"rig": {"sensors": []}}),
            },
            "version": 1,
        },
    )
    _write_parquet(clipgt / "calibration_estimate.parquet", calibration_rows)

    output = build_alpasim_clipgt_usdz(
        tmp_path / "scene.usdz",
        clipgt_dir=clipgt,
        initial_frame_path=_make_jpeg(tmp_path),
    )

    assert output.is_file()


@pytest.mark.parametrize("required_file", _REQUIRED_MAP_FILES)
def test_build_requires_alpasim_map_tables(tmp_path: Path, required_file: str) -> None:
    clipgt = _make_clipgt_dir(tmp_path)
    (clipgt / required_file).unlink()
    with pytest.raises(
        FileNotFoundError, match="ClipGT directory is missing required parquet"
    ) as error:
        build_alpasim_clipgt_usdz(
            tmp_path / "scene.usdz",
            clipgt_dir=clipgt,
            initial_frame_path=_make_jpeg(tmp_path),
        )
    assert required_file in str(error.value)


def test_build_is_byte_deterministic_with_checkpoint_free_date(tmp_path: Path) -> None:
    clipgt = _make_clipgt_dir(tmp_path)
    first_frame = _make_jpeg(tmp_path)
    first = build_alpasim_clipgt_usdz(
        tmp_path / "first.usdz",
        clipgt_dir=clipgt,
        initial_frame_path=first_frame,
    )
    second = build_alpasim_clipgt_usdz(
        tmp_path / "second.usdz",
        clipgt_dir=clipgt,
        initial_frame_path=first_frame,
    )

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        metadata = yaml.safe_load(archive.read("metadata.yaml"))
    assert metadata["training_date"] == "1970-01-01"


def test_build_requires_exactly_one_initial_image_source(tmp_path: Path) -> None:
    clipgt = _make_clipgt_dir(tmp_path)
    first_frame = _make_jpeg(tmp_path)
    with pytest.raises(ValueError, match="Exactly one"):
        build_alpasim_clipgt_usdz(tmp_path / "none.usdz", clipgt_dir=clipgt)
    with pytest.raises(ValueError, match="Exactly one"):
        build_alpasim_clipgt_usdz(
            tmp_path / "both.usdz",
            clipgt_dir=clipgt,
            initial_frame_path=first_frame,
            initial_video_path=first_frame,
        )


def test_build_rejects_non_jpeg_initial_frame(tmp_path: Path) -> None:
    clipgt = _make_clipgt_dir(tmp_path)
    png = tmp_path / "first.png"
    buffer = io.BytesIO()
    Image.new("RGB", (32, 18), color="red").save(buffer, format="PNG")
    png.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="JPEG bytes"):
        build_alpasim_clipgt_usdz(
            tmp_path / "scene.usdz",
            clipgt_dir=clipgt,
            initial_frame_path=png,
        )


def test_ground_mesh_uses_pose_heading_and_expands_to_scene_geometry() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.2, 0.0]],
        dtype=np.float32,
    )
    quaternions = np.asarray(
        [[0.0, 0.0, 0.0, 1.0]] * len(positions),
        dtype=np.float32,
    )
    coverage_points = np.asarray([[0.0, 160.0, 0.0]], dtype=np.float64)

    vertices, _faces = load_mesh_vf(
        _ground_ribbon_mesh(positions, quaternions, coverage_points)
    )

    assert vertices[:, 0].min() < -160.0
    assert vertices[:, 0].max() > 160.0
    assert vertices[:, 1].min() < -169.0
    assert vertices[:, 1].max() > 169.0
