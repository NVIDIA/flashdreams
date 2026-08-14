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

"""Architecture regressions for the Crazy Robotaxi application boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import crazy_robotaxi
import omnidreams_game_engine as interactive_drive
import pytest
from omnidreams_game_engine.types import SceneBundle

pytestmark = pytest.mark.ci_cpu

_SHARED_RUNTIME_MODULES = (
    "app.py",
    "application.py",
    "config.py",
    "input/keyboard.py",
    "runtime/loop.py",
    "scene_loader.py",
    "slangpy_hud_presenter.py",
    "streaming_presenter.py",
    "video_model/chunk_pipeline.py",
)


def test_shared_runtime_does_not_import_crazy_robotaxi() -> None:
    """Keep app policy behind generic seams; CLI/demo are composition roots."""
    package_root = Path(interactive_drive.__file__).resolve().parent
    violations: list[str] = []
    for relative_path in _SHARED_RUNTIME_MODULES:
        path = package_root / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_modules: tuple[str, ...]
            if isinstance(node, ast.Import):
                imported_modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules = (node.module,)
            else:
                continue
            if any("crazy_robotaxi" in module for module in imported_modules):
                violations.append(relative_path)
    assert violations == []


@pytest.mark.parametrize("package", [interactive_drive, crazy_robotaxi])
def test_standalone_packages_do_not_import_interactive_drive(package: object) -> None:
    """Keep the standalone source independent of the enterprise demo."""
    package_file = getattr(package, "__file__")
    package_root = Path(package_file).resolve().parent
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            else:
                continue
            if any(
                module.startswith("omnidreams.interactive_drive") for module in modules
            ):
                violations.append(str(path.relative_to(package_root)))
    assert violations == []


def test_shared_scene_bundle_has_no_taxi_navigation_fields() -> None:
    field_names = SceneBundle.__dataclass_fields__
    assert "reference_route_world" not in field_names
    assert "navigation_routes_world" not in field_names
