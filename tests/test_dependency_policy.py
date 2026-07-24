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

"""Enforce dependency policies shared by every workspace package."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import tomli as tomllib

pytestmark = pytest.mark.ci_cpu

_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_OPENCV_DISTRIBUTIONS = frozenset(
    {
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
        "opencv-python",
    }
)
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _workspace_pyprojects() -> list[Path]:
    root_pyproject = _ROOT / "pyproject.toml"
    root_config = tomllib.loads(root_pyproject.read_text(encoding="utf-8"))
    member_patterns = root_config["tool"]["uv"]["workspace"]["members"]
    members = {
        member / "pyproject.toml"
        for pattern in member_patterns
        for member in _ROOT.glob(pattern)
        if (member / "pyproject.toml").is_file()
    }
    return [root_pyproject, *sorted(members)]


def _requirement_strings(config: dict[str, Any]) -> Iterable[tuple[str, str]]:
    project = config.get("project", {})
    if isinstance(project, dict):
        for requirement in project.get("dependencies", []):
            if isinstance(requirement, str):
                yield "project.dependencies", requirement

        optional_dependencies = project.get("optional-dependencies", {})
        if isinstance(optional_dependencies, dict):
            for extra, requirements in optional_dependencies.items():
                for requirement in requirements:
                    if isinstance(requirement, str):
                        yield f"project.optional-dependencies.{extra}", requirement

    dependency_groups = config.get("dependency-groups", {})
    if isinstance(dependency_groups, dict):
        for group, requirements in dependency_groups.items():
            for requirement in requirements:
                if isinstance(requirement, str):
                    yield f"dependency-groups.{group}", requirement

    build_system = config.get("build-system", {})
    if isinstance(build_system, dict):
        for requirement in build_system.get("requires", []):
            if isinstance(requirement, str):
                yield "build-system.requires", requirement


def _normalized_requirement_name(requirement: str) -> str | None:
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        return None
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def test_workspace_uses_only_headless_opencv() -> None:
    pyprojects = _workspace_pyprojects()
    assert len(pyprojects) > 1, (
        "expected the root project and at least one workspace member"
    )

    violations: list[str] = []
    for pyproject in pyprojects:
        config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        for section, requirement in _requirement_strings(config):
            name = _normalized_requirement_name(requirement)
            if name in _FORBIDDEN_OPENCV_DISTRIBUTIONS:
                relative_path = pyproject.relative_to(_ROOT)
                violations.append(f"{relative_path} [{section}]: {requirement}")

    assert not violations, (
        "Workspace packages must use only opencv-python-headless; found:\n"
        + "\n".join(f"  {violation}" for violation in violations)
    )
