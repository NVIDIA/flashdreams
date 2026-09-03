# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for Waypoint's shared Action2V binding."""

from pathlib import Path

import pytest
import tomli as tomllib
from action2v import Action2VApplication, ActionSnapshot
from waypoint import WaypointControl
from waypoint.apps.action2v.adapter import WaypointApplication
from waypoint.impl.input_mapping import WaypointActionMapper

pytestmark = pytest.mark.ci_cpu


def test_waypoint_application_uses_shared_action2v_shell() -> None:
    """Keep v2 application lifecycle outside the model integration."""
    assert isinstance(WaypointApplication(), Action2VApplication)


def test_waypoint_package_registers_action2v_slug() -> None:
    """Expose the shared application under the Action2V slug."""
    package_root = Path(__file__).parents[1]
    with (package_root / "pyproject.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    entry_points = manifest["project"]["entry-points"]["flashdreams.applications_v2"]
    assert entry_points == {
        "action2v-waypoint-1-5-1b": "waypoint.apps.action2v.adapter:create_app",
    }
    assert manifest["project"]["name"] == "flashdreams-waypoint"
    assert "flashdreams-action2v" in manifest["project"]["dependencies"]
    assert "flashdreams-waypoint" not in manifest["project"]["dependencies"]
    assert (package_root / "config.py").is_file()
    assert (package_root / "impl").is_dir()
    assert (package_root / "apps" / "action2v" / "adapter.py").is_file()
    assert manifest["tool"]["setuptools"]["packages"] == [
        "waypoint",
        "waypoint.apps",
        "waypoint.apps.action2v",
        "waypoint.impl",
        "waypoint.impl.transformer",
    ]
    assert manifest["tool"]["setuptools"]["package-dir"] == {"waypoint": "."}
    assert manifest["tool"]["setuptools"]["package-data"] == {
        "waypoint.apps.action2v": ["assets/*.json"]
    }
    assert (
        package_root / "apps" / "action2v" / "assets" / "example_controls.json"
    ).is_file()
    repository_root = package_root.parents[1]
    assert not (
        repository_root / "integrations" / "waypoint" / "pyproject.toml"
    ).exists()


def test_waypoint_mapper_keeps_checkpoint_semantics_model_owned() -> None:
    """Map normalized shared input only at the integration boundary."""
    mapper = WaypointActionMapper(
        video_width=100,
        video_height=50,
        mouse_sensitivity=2.0,
    )
    assert mapper(
        ActionSnapshot(
            keys=frozenset({"W", "SHIFT"}),
            mouse_buttons=frozenset({2}),
            mouse_dx=0.25,
            mouse_dy=-0.5,
            wheel_y=-0.25,
        )
    ) == WaypointControl(
        buttons=frozenset({87, 0x10, 0x02}),
        mouse_dx=50.0,
        mouse_dy=-50.0,
        scroll_wheel=-1,
    )
