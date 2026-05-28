# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from omnidreams.interactive_drive.config import RasterConfig
from omnidreams.interactive_drive.scene_fixture import build_synthetic_scene_usdz
from omnidreams.interactive_drive.scene_loader import load_scene_bundle
from omnidreams.interactive_drive.synthetic_scene import (
    build_default_synthetic_scene,
    build_synthetic_scene_to_temp,
)
from PIL import Image


def _make_solid_rgb(tmp_path: Path, color: tuple[int, int, int]) -> Path:
    img = Image.new("RGB", (320, 180), color=color)
    out = tmp_path / "starter.png"
    img.save(out)
    return out


def test_synthetic_scene_round_trip_with_overrides(tmp_path: Path) -> None:
    """Scene loader must accept a synthetic USDZ built with caller overrides."""
    initial_rgb = _make_solid_rgb(tmp_path, color=(120, 60, 200))
    custom_prompt = "A custom synthetic prompt for the China demo."

    scene_path = build_default_synthetic_scene(
        tmp_path / "synthetic.usdz",
        initial_rgb_path=initial_rgb,
        prompt=custom_prompt,
        length_km=0.02,
    )
    bundle = load_scene_bundle(
        scene_path=scene_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=640, height=352),
    )

    # The custom prompt must round-trip end-to-end.
    assert bundle.prompt == custom_prompt
    # The initial frame should be resized to RasterConfig but should
    # carry the colour we put in (purple-ish; loader's BILINEAR resize
    # will not change the dominant channel ordering meaningfully).
    assert bundle.initial_rgb.shape == (352, 640, 3)
    mean_rgb = bundle.initial_rgb.reshape(-1, 3).mean(axis=0)
    # Roughly purple: red ~120, green ~60, blue ~200 (within ±5 after
    # PNG round-trip + bilinear resize on a uniform image).
    assert abs(mean_rgb[0] - 120) < 5
    assert abs(mean_rgb[1] - 60) < 5
    assert abs(mean_rgb[2] - 200) < 5


def test_synthetic_scene_default_image_round_trip(tmp_path: Path) -> None:
    """No override path must still produce a loadable USDZ (debug gradient)."""
    scene_path = build_default_synthetic_scene(
        tmp_path / "default.usdz", length_km=0.02
    )
    bundle = load_scene_bundle(
        scene_path=scene_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=320, height=176),
    )
    assert bundle.initial_rgb.shape == (176, 320, 3)
    assert "Synthetic" in bundle.prompt


def test_build_synthetic_scene_usdz_validates_initial_rgb_shape(
    tmp_path: Path,
) -> None:
    """A bogus initial_rgb shape must raise rather than corrupt the USDZ."""
    bad = np.zeros((10, 10), dtype=np.uint8)  # missing channel dim
    with pytest.raises(ValueError, match="initial_rgb must be"):
        build_synthetic_scene_usdz(tmp_path / "bad.usdz", initial_rgb=bad)


def test_synthetic_scene_to_temp_creates_loadable_file() -> None:
    """The temp-file builder must produce a path the scene loader accepts.

    Uses ``length_km=0.02`` so the test stays a sub-second smoke check
    rather than paying for the runtime default's 30 000-frame trajectory.
    """
    scene_path = build_synthetic_scene_to_temp(length_km=0.02)
    assert scene_path.exists()
    bundle = load_scene_bundle(
        scene_path=scene_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=320, height=176),
    )
    assert bundle.initial_rgb.shape == (176, 320, 3)


def test_initial_rgb_dtype_coercion(tmp_path: Path) -> None:
    """Floating-point ``[0..255]`` arrays should be clipped + cast to uint8."""
    rgb_float = np.full((50, 60, 3), fill_value=200.7, dtype=np.float32)
    scene_path = build_synthetic_scene_usdz(
        tmp_path / "float.usdz",
        initial_rgb=rgb_float,
    )
    # Round-trip through the loader to confirm the embedded PNG is valid.
    bundle = load_scene_bundle(
        scene_path=scene_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=120, height=80),
    )
    assert bundle.initial_rgb.dtype == np.uint8
    # 200.7 -> 200 after clip+cast; allow ±2 for resize.
    assert abs(int(bundle.initial_rgb.mean()) - 200) < 3


def test_length_frames_extends_trajectory(tmp_path: Path) -> None:
    """Bumping ``length_frames`` must produce a proportionally longer trajectory."""
    short = build_synthetic_scene_usdz(tmp_path / "short.usdz", length_frames=60)
    long = build_synthetic_scene_usdz(tmp_path / "long.usdz", length_frames=1200)

    short_bundle = load_scene_bundle(
        scene_path=short,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=160, height=88),
    )
    long_bundle = load_scene_bundle(
        scene_path=long,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=160, height=88),
    )

    short_metadata = dict(short_bundle.metadata)
    long_metadata = dict(long_bundle.metadata)
    short_span = (
        short_metadata["time_range"]["end"] - short_metadata["time_range"]["start"]
    )
    long_span = (
        long_metadata["time_range"]["end"] - long_metadata["time_range"]["start"]
    )
    # 20x as many frames -> ~20x the trajectory time span.
    assert long_span > 15 * short_span


def test_length_frames_validates_minimum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="length_frames"):
        build_synthetic_scene_usdz(tmp_path / "tiny.usdz", length_frames=1)


def test_build_synthetic_scene_to_temp_honours_length_km() -> None:
    """``length_km=0.02`` should produce a scene with ~2 s of trajectory time
    (20 m of road @ 10 m/s)."""
    scene_path = build_synthetic_scene_to_temp(length_km=0.02)
    bundle = load_scene_bundle(
        scene_path=scene_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=160, height=88),
    )
    metadata = dict(bundle.metadata)
    span_us = metadata["time_range"]["end"] - metadata["time_range"]["start"]
    # 20 m / 10 m/s = 2 s = 60 frames at 30 fps; allow a few-frame jitter
    # for end-frame indexing and rounding.
    assert 1_900_000 <= span_us <= 2_100_000


def test_off_road_clutter_extends_well_beyond_road(tmp_path: Path) -> None:
    """Off-road poles should scatter past the road boundary so the HDMap
    stays non-empty when the ego strays off the lane.
    """
    scene_path = build_default_synthetic_scene(tmp_path / "clutter.usdz", length_km=0.5)
    bundle = load_scene_bundle(
        scene_path=scene_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=160, height=88),
    )
    pole_layers = [layer for layer in bundle.line_layers if layer.layer_name == "poles"]
    assert pole_layers, "synthetic scene must publish a 'poles' line layer"
    # ``segments_world`` is (N, 2, 3); take the absolute Y of every pole base.
    bases_y = np.abs(pole_layers[0].segments_world[:, 0, 1])
    # Periodic streetlamp poles sit at exactly 9.5 m. Off-road poles sit
    # in a [15, 100] m band, so we should see plenty of |y| > 15 m.
    far_field_count = int((bases_y > 15.0).sum())
    near_road_count = int((bases_y <= 9.5 + 0.1).sum())
    assert far_field_count > 50, f"expected off-road poles, got {far_field_count}"
    assert near_road_count > 0, (
        f"expected periodic streetlamp poles, got {near_road_count}"
    )


def test_off_road_clutter_is_deterministic(tmp_path: Path) -> None:
    """Two builds at the same length must yield identical pole point clouds."""
    a = build_default_synthetic_scene(tmp_path / "a.usdz", length_km=0.5)
    b = build_default_synthetic_scene(tmp_path / "b.usdz", length_km=0.5)
    raster = RasterConfig(width=160, height=88)
    bundle_a = load_scene_bundle(
        scene_path=a,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=raster,
    )
    bundle_b = load_scene_bundle(
        scene_path=b,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=raster,
    )
    poles_a = next(
        layer for layer in bundle_a.line_layers if layer.layer_name == "poles"
    )
    poles_b = next(
        layer for layer in bundle_b.line_layers if layer.layer_name == "poles"
    )
    np.testing.assert_array_equal(poles_a.segments_world, poles_b.segments_world)


def test_initial_rgb_in_memory_pil_load(tmp_path: Path) -> None:
    """End-to-end: PIL-supported in-memory image should round-trip."""
    src = Image.new("RGB", (300, 200), color=(10, 220, 30))
    buf = io.BytesIO()
    src.save(buf, format="PNG")
    buf.seek(0)
    starter = tmp_path / "green.png"
    starter.write_bytes(buf.getvalue())

    scene_path = build_default_synthetic_scene(
        tmp_path / "green.usdz",
        initial_rgb_path=starter,
        length_km=0.02,
    )
    bundle = load_scene_bundle(
        scene_path=scene_path,
        camera_name="camera_front_wide_120fov",
        variant="default",
        prompt_override=None,
        raster=RasterConfig(width=160, height=88),
    )
    mean = bundle.initial_rgb.reshape(-1, 3).mean(axis=0)
    assert abs(mean[1] - 220) < 5
    assert abs(mean[0] - 10) < 5
    assert abs(mean[2] - 30) < 5
