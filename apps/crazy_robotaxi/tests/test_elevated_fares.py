# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import yaml
from crazy_robotaxi.navigation import TaxiNavigationMap
from crazy_robotaxi.scene import load_scene_data
from omnidreams_game_engine.config import RasterConfig
from omnidreams_game_engine.scene import SceneRequest, load_scene
from omnidreams_game_engine.simulation.ground_snap import GroundSnapper

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("elevation", [-4.0, 8.0])
def test_parking_fares_and_passengers_are_on_the_lot_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, elevation: float
) -> None:
    fixture = Path(__file__).parent / "maps/parking_driveway.robotaxi.yaml"
    doc = yaml.safe_load(fixture.read_text())
    doc["schema_version"] = 2
    driveway = next(node for node in doc["nodes"] if node["id"] == "lot_driveway")
    driveway["pose"]["z_m"] = elevation
    source = tmp_path / fixture.name
    source.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    scene = load_scene(SceneRequest(map_path=source), RasterConfig(width=64, height=32))
    data = load_scene_data(scene)
    nav = TaxiNavigationMap(data.navigation_lanes)
    fares = nav.sample_fare_regions(data.fare_regions, 15, np.random.default_rng(42))
    parking = [fare for fare in fares if fare.element_id == "parking_lot"]
    assert parking
    assert (
        scene.ground_mesh_vertices is not None and scene.ground_mesh_faces is not None
    )
    terrain = GroundSnapper(scene.ground_mesh_vertices, scene.ground_mesh_faces)
    for fare in parking:
        assert fare.passenger_xyz_m is not None
        for point in (fare.xyz_m, fare.passenger_xyz_m):
            assert point[2] == pytest.approx(elevation)
            assert terrain.ground_height_at(*point) == pytest.approx(point[2], abs=1e-4)
