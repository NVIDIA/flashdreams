# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import io
import zipfile

import pytest
from omnidreams.interactive_drive._sample_assets import SAMPLE_SCENE
from omnidreams.interactive_drive.colors import BBOX_V3_COLORS
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.scene_fixture import build_synthetic_scene_usdz
from omnidreams.interactive_drive.scene_loader import (
    _discover_prompts,
    load_scene_bundle,
)


def test_usdz_prompt_discovery_accepts_legacy_numeric_suffix() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("prompt1.txt", "legacy one")
        zf.writestr("prompt_2.txt", "canonical two")
        zf.writestr("promptnight.txt", "ignored")

    archive.seek(0)
    with zipfile.ZipFile(archive, "r") as zf:
        prompts = _discover_prompts(zf)

    assert prompts["default"] == "legacy one"
    assert prompts["1"] == "legacy one"
    assert prompts["2"] == "canonical two"
    assert "night" not in prompts


def test_load_scene_bundle_from_extracted_scene_directory(tmp_path) -> None:
    archive_path = build_synthetic_scene_usdz(tmp_path / "scene.usdz")
    scene_dir = tmp_path / "scene"
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(scene_dir)

    bundle = load_scene_bundle(
        scene_path=scene_dir,
        camera_name="camera_front_wide_120fov",
        variant="1",
        prompt_override=None,
        raster=RasterConfig(width=320, height=176),
    )

    assert bundle.scene_path == scene_dir
    assert bundle.scene_id == "synthetic-test-scene"
    assert bundle.selected_camera.logical_name == "camera_front_wide_120fov"
    assert bundle.initial_rgb.shape == (176, 320, 3)
    assert bundle.prompt == "Synthetic prompt variant 1."
    assert len(bundle.vehicle_bbox_tracks) > 0


# Opportunistic: exercises the real USDZ loader, so this test is silently
# skipped on machines where ``prepare.py`` hasn't fetched the production asset.
@pytest.mark.skipif(
    not SAMPLE_SCENE.exists(),
    reason="sample scene is not available on this workstation",
)
def test_load_scene_bundle_from_real_usdz() -> None:
    bundle = load_scene_bundle(
        scene_path=SAMPLE_SCENE,
        camera_name="camera_front_wide_120fov",
        variant="1",
        prompt_override=None,
        raster=RasterConfig(width=640, height=352),
    )

    assert bundle.scene_id.startswith("clipgt-")
    assert bundle.selected_camera.logical_name == "camera_front_wide_120fov"
    assert bundle.initial_rgb.shape == (352, 640, 3)
    assert bundle.initial_timestamp_us > 0
    assert len(bundle.line_layers) > 0
    assert any(layer.color_rgba == (1.0, 1.0, 0.0, 1.0) for layer in bundle.line_layers)
    assert any(
        layer.layer_name == "traffic_signs" and len(layer.triangles_world) > 0
        for layer in bundle.triangle_layers
    )
    assert any(
        layer.layer_name == "crosswalks" and len(layer.polygons_world) > 0
        for layer in bundle.polygon_layers
    )
    assert len(bundle.vehicle_bbox_tracks) > 0
    sample_track = bundle.vehicle_bbox_tracks[0]
    assert sample_track.object_type in BBOX_V3_COLORS
    assert (
        sample_track.interpolate_at_timestamp(bundle.initial_timestamp_us) is not None
    )
