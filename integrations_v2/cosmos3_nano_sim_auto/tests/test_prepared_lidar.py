# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared-scene metadata validation, the renderer gate and renderer cases.

The geometry/projection operations are covered by ``test_lidar_geometry.py`` and
``test_lidar_projection.py``. Rendering is enabled; gate tests switch
``IMPLEMENTATION_AVAILABLE`` off to keep the early-failure path covered.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    _lidar_geometry as geometry,
)
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    _lidar_projection as projection,
)
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    prepared_lidar,
)

pytestmark = pytest.mark.ci_cpu

UNAVAILABLE = "Prepared-scene LiDAR rendering is unavailable"


@pytest.fixture
def disabled(monkeypatch):
    """Switch rendering off, as a build would, to cover the early-failure path."""
    monkeypatch.setattr(projection, "IMPLEMENTATION_AVAILABLE", False)


def lidar_data(camera="front"):
    """Create synthetic metadata without sampling or projecting any points."""
    return {
        "settings": dict(prepared_lidar.SETTINGS),
        "coordinates": "metres; world and ego FLU",
        "profile": {
            "row_elevations_rad": np.linspace(0.25, -0.45, 128).tolist(),
            "row_azimuth_offsets_rad": [0.01] * 128,
        },
        "reference_camera": camera,
        "sensor_to_ego": np.eye(4).tolist(),
        "camera_to_ego": np.eye(4).tolist(),
        "provenance": dict(
            source_url="synthetic",
            source_revision="fixture-v1",
            source_sha256="synthetic",
            mount_description="identity mount",
        ),
        "layers": {name: [] for name in geometry.DEFAULT_MINIMAP_TYPES},
        "objects": [],
    }


def test_metadata_loading_and_rendering_is_enabled():
    loaded = prepared_lidar.load(lidar_data(), dict(camera_order=["front"], frames=4))
    assert loaded.model.n_columns == 1800
    assert loaded.model.spinning_frequency_hz == 10
    np.testing.assert_array_equal(loaded.sensor_to_ego, np.eye(4))
    assert geometry.IMPLEMENTATION_AVAILABLE and projection.IMPLEMENTATION_AVAILABLE
    prepared_lidar.require_implementation()
    assert loaded.object_points(0).shape == (0, 3)


@pytest.mark.parametrize(
    "fault", ["mount", "calibration", "provenance", "layers", "actor"]
)
def test_metadata_validation_remains_active(fault):
    data = lidar_data()
    if fault == "mount":
        data["sensor_to_ego"][0][0] = -1
    elif fault == "calibration":
        data["profile"]["row_elevations_rad"][0] = float("nan")
    elif fault == "provenance":
        data["provenance"]["source_revision"] = ""
    elif fault == "layers":
        del data["layers"]["poles"]
    else:
        data["objects"] = [
            {
                "id": "a",
                "frames": [0, 0],
                "lwh": [[1, 1, 1]] * 2,
                "poses": [np.eye(4).tolist()] * 2,
            }
        ]
    with pytest.raises(ValueError):
        prepared_lidar.load(data, dict(camera_order=["front"], frames=4))


@pytest.mark.parametrize("module", [geometry, projection])
def test_either_half_disabled_disables_rendering(monkeypatch, module):
    monkeypatch.setattr(module, "IMPLEMENTATION_AVAILABLE", False)
    with pytest.raises(NotImplementedError, match=UNAVAILABLE):
        prepared_lidar.require_implementation()


@pytest.mark.parametrize(
    "options", [{}, {"optimized": True}, {"cull": True}, {"settle": True}]
)
def test_renderer_rejects_before_mapping_sampling_or_device_upload(
    monkeypatch, disabled, options
):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled renderer performed work")

    scene = SimpleNamespace(lidar=object(), frame_mapping=forbidden)
    monkeypatch.setattr(torch, "as_tensor", forbidden)
    monkeypatch.setattr(geometry, "minimap_points", forbidden)
    with pytest.raises(NotImplementedError, match=UNAVAILABLE):
        prepared_lidar.Renderer(scene, device="cuda:0", **options)


@pytest.mark.parametrize("rank", [None, 0, 3])
def test_live_stream_rejects_before_cached_renderer_codec_or_collectives(
    monkeypatch, disabled, rank
):
    from cosmos3_nano_sim_auto.apps.interactive_drive import lidar

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled stream used cached state, a checkpoint or a collective")

    prepared = SimpleNamespace(lidar=object())
    scene = SimpleNamespace(metadata=dict(prepared_scene=prepared))
    parallel = None if rank is None else SimpleNamespace(world_size=4, rank=rank)
    monkeypatch.setattr(
        lidar, "_RENDERER", [prepared, (torch.device("cpu"), False, False), forbidden]
    )
    monkeypatch.setattr(lidar, "_codec", forbidden)
    monkeypatch.setattr(lidar, "lidar_ranks", forbidden)
    with pytest.raises(NotImplementedError, match=UNAVAILABLE):
        lidar.make_live_lidar(
            scene,
            checkpoint="unused",
            patch_dim=128,
            device="cpu",
            parallel=cast(Any, parallel),
        )


@pytest.mark.parametrize("preloaded", [False, True])
def test_backend_lidar_prewarm_rejects_before_loading_weights(disabled, preloaded):
    from cosmos3_nano_sim_auto.apps.interactive_drive.backend import (
        MultiViewRenderBackend,
    )

    backend = object.__new__(MultiViewRenderBackend)
    backend._settings = SimpleNamespace(lidar_checkpoint=Path("unused.pt"))
    backend._network = object() if preloaded else None
    with pytest.raises(NotImplementedError, match=UNAVAILABLE):
        backend.warmup_model()


def test_lidar_prewarm_with_resident_network_is_available():
    from cosmos3_nano_sim_auto.apps.interactive_drive.backend import (
        MultiViewRenderBackend,
    )

    backend = object.__new__(MultiViewRenderBackend)
    backend._settings = SimpleNamespace(lidar_checkpoint=Path("unused.pt"))
    backend._network = object()
    backend.warmup_model()


def test_rgb_prewarm_with_resident_network_remains_available():
    from cosmos3_nano_sim_auto.apps.interactive_drive.backend import (
        MultiViewRenderBackend,
    )

    backend = object.__new__(MultiViewRenderBackend)
    backend._settings = SimpleNamespace(lidar_checkpoint=None)
    backend._network = object()
    backend.warmup_model()


# Renderer cases (I09, I11, I12, I15) through the real prepared_lidar.Renderer.


def wall(x, half_y, half_z, y=0.0):
    """Vertical rectangle facing the sensor at distance ``x`` (a polygon layer)."""
    return [
        [x, y - half_y, -half_z],
        [x, y + half_y, -half_z],
        [x, y + half_y, half_z],
        [x, y - half_y, half_z],
    ]


def renderer(monkeypatch, data, mapping=(0, 0, 0, 1, 1, 1, 2, 2, 2)):
    loaded = prepared_lidar.load(data, dict(camera_order=["front"], frames=3))
    scene = SimpleNamespace(lidar=loaded, frame_mapping=lambda *_: np.array(mapping))
    return prepared_lidar.Renderer(scene, device="cpu"), loaded.model


def cell(model, elevation, azimuth):
    row = int(np.argmin(np.abs(model.row_elevations_rad - elevation)))
    column = int((np.pi - azimuth) // (2 * np.pi / model.n_columns)) % model.n_columns
    return row, column


def test_I09_empty_scene_renders_zeros(monkeypatch):
    render, _ = renderer(monkeypatch, lidar_data())
    assert render.density == geometry.DEFAULT_DENSITY_M
    out = render(0, np.eye(4), np.eye(4))
    assert out.shape == (3, 128, 1800) and out.dtype == torch.float32
    assert not out.any()


def test_point_cap_coarsens_the_whole_map_once(monkeypatch, caplog):
    monkeypatch.setattr(geometry, "MAX_SCENE_POINTS", 2_000)
    data = lidar_data()
    data["layers"]["road_markings"] = [wall(20, 1, 1), wall(-20, 1, 1)]
    render, _ = renderer(monkeypatch, data)
    assert render.density > geometry.DEFAULT_DENSITY_M
    assert 0 < len(render.static) <= geometry.MAX_SCENE_POINTS
    assert render.static[:, 0].min() == -20 and render.static[:, 0].max() == 20
    assert "spacing for the entire scene" in caplog.text

    def forbidden(*args, **kwargs):
        pytest.fail("Resampled the map during rendering")

    monkeypatch.setattr(geometry, "minimap_points", forbidden)
    expected = render(0, np.eye(4), np.eye(4))
    assert expected[2].any()
    assert torch.equal(expected, render(1, np.eye(4), np.eye(4)))


@pytest.mark.parametrize("limit", ["scene", "primitive"])
def test_point_cap_reserves_for_large_actors_in_later_frames(monkeypatch, limit):
    monkeypatch.setattr(
        geometry,
        "MAX_SCENE_POINTS" if limit == "scene" else "MAX_PRIMITIVE_POINTS",
        20_000 if limit == "scene" else 5_000,
    )
    data = lidar_data()
    pose = np.eye(4)
    pose[0, 3] = 15
    data["objects"] = [
        dict(
            id="car",
            frames=[0, 2],
            lwh=[[1, 1, 1], [8, 4, 3]],
            poses=[pose.tolist()] * 2,
        )
    ]
    render, _ = renderer(monkeypatch, data)
    assert render.density > geometry.DEFAULT_DENSITY_M
    capacity = render.data.actor_point_capacity(render.density)
    assert capacity <= geometry.MAX_SCENE_POINTS
    for frame in (0, 1, 2):
        points = render.data.object_points(frame, density=render.density)
        assert len(points) <= capacity
        assert len(points) <= geometry.MAX_PRIMITIVE_POINTS
        assert len(render.static) + len(points) <= geometry.MAX_SCENE_POINTS
    out = render(2, np.eye(4), np.eye(4))
    assert out.shape == (3, 128, 1800) and torch.isfinite(out).all()


def test_point_cap_does_not_hide_invalid_geometry(monkeypatch):
    data = lidar_data()
    data["layers"]["road_markings"] = [[[20, 0, 0], [20, 1, 1], [20, 0, 1], [20, 1, 0]]]
    with pytest.raises(ValueError, match="self-intersects"):
        renderer(monkeypatch, data)


def test_point_cap_retries_are_bounded_for_unrepresentable_input(monkeypatch):
    monkeypatch.setattr(geometry, "MAX_SCENE_POINTS", 1)
    data = lidar_data()
    data["layers"]["poles"] = [[[10, 0, 0], [10, 0, 1]]]
    with pytest.raises(geometry.PointLimitError):
        renderer(monkeypatch, data)


def test_empty_scene_fits_a_single_point_primitive_cap(monkeypatch):
    monkeypatch.setattr(geometry, "MAX_PRIMITIVE_POINTS", 1)
    render, _ = renderer(monkeypatch, lidar_data())
    assert render.density == geometry.DEFAULT_DENSITY_M
    assert not len(render.static)
    assert not render(0, np.eye(4), np.eye(4)).any()


def test_I12_nearest_return_occludes_before_training_validity(monkeypatch):
    data = lidar_data()
    data["layers"]["road_markings"] = [
        wall(20, 4, 4),  # far wall, partly hidden
        wall(2, 0.1, 0.1),  # 2 m occluder in front of it
        wall(-5, 1, 1),  # 5 m wall behind the sensor
        wall(100, 5, 5, y=0),  # 100 m wall: invalid
    ]
    data["layers"]["road_markings"][3] = [[x, 100.0, z] for x, _, z in wall(1, 5, 5)]
    render, model = renderer(monkeypatch, data)
    ranges, control, valid = render(0, np.eye(4), np.eye(4))
    assert not control.any()
    hidden = cell(model, 0.0, 0.01)
    assert ranges[hidden] == 0 and valid[hidden] == 0  # 2 m return wins, then invalid
    visible = cell(model, 0.05, 0.12)
    assert valid[visible] == 1 and 20 <= ranges[visible] < 20.5
    behind = cell(model, 0.0, np.pi - 0.01)
    assert valid[behind] == 1 and 5 <= ranges[behind] < 5.05
    left = cell(model, 0.0, np.pi / 2)
    assert valid[left] == 0 and ranges[left] == 0
    assert ranges.max() < 100 and set(valid.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(ranges > 0, valid.bool())


def test_I11_actors_follow_source_frames_and_hold_after_scene_end(monkeypatch):
    data = lidar_data()
    ahead, behind = np.eye(4), np.eye(4)
    ahead[0, 3], behind[0, 3] = 10, -10
    data["objects"] = [
        dict(
            id="car",
            frames=[0, 2],
            lwh=[[2, 1, 1], [3, 2, 1]],
            poses=[ahead.tolist(), behind.tolist()],
        )
    ]
    render, model = renderer(monkeypatch, data)
    # actors are box wireframes: aim at the vertical edges nearest the sensor
    forward = cell(model, 0.0, np.arctan2(0.5, 9))
    rear = cell(model, 0.0, np.pi - np.arctan2(1, 8.5))
    lidar = render.data
    assert len(lidar.object_points(0)) and len(lidar.object_points(2))
    assert lidar.object_points(1).shape == (0, 3)  # gap frame: no ghost
    first = render(0, np.eye(4), np.eye(4))  # source frame 0
    assert first[2][forward] == 1 and 8.9 < first[0][forward] < 9.1
    assert first[2][rear] == 0
    gap = render(1, np.eye(4), np.eye(4))  # source frame 1
    assert not gap.any()
    for index in (2, 50):  # frame 2, then held beyond the scene end
        out = render(index, np.eye(4), np.eye(4))
        assert out[2][rear] == 1 and 8.4 < out[0][rear] < 8.6
        assert out[2][forward] == 0


@pytest.mark.parametrize(
    "options", [{}, {"optimized": True}, {"cull": True}, {"settle": True}]
)
def test_I15_renderer_options_agree_on_static_and_moving_sweeps(monkeypatch, options):
    data = lidar_data()
    data["layers"]["road_markings"] = [
        wall(20, 4, 4),
        wall(-12, 6, 3),
        wall(2, 0.1, 0.1),
    ]
    data["layers"]["poles"] = [[[8, 3, -1], [8, 3, 1]]]
    base, _ = renderer(monkeypatch, data)
    render, _ = renderer(monkeypatch, data)
    for name, value in options.items():
        setattr(render, name, value)
    moved = np.eye(4)
    moved[:3, :3] = [
        [np.cos(0.05), -np.sin(0.05), 0],
        [np.sin(0.05), np.cos(0.05), 0],
        [0, 0, 1],
    ]
    moved[:3, 3] = [1.5, 0, 0]
    for first, last in ((np.eye(4), np.eye(4)), (np.eye(4), moved), (moved, np.eye(4))):
        expected, actual = base(0, first, last), render(0, first, last)
        assert torch.equal(expected[2], actual[2])
        torch.testing.assert_close(actual[0], expected[0], atol=1e-3, rtol=0)
