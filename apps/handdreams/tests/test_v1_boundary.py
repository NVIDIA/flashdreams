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

"""Strict isolation of the HandDreams application from model and legacy packages."""

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

pytestmark = pytest.mark.ci_cpu

_PROJECT = Path(__file__).resolve().parents[1]
_REPOSITORY = _PROJECT.parents[1]
_PACKAGE = _PROJECT / "handdreams"
_INPUT_PROJECT = _PROJECT / "controller_input"
_INPUT_PACKAGE = _INPUT_PROJECT / "flashdreams_controller_input"
_FORBIDDEN_ROOTS = (
    "cosmos_interactive_internal",
    "cosmos_interactive_hands",
    "cosmos_interactive_camera",
    "internal",
    "daydreams",
    "cosmos3_nano_sim_bimanual",
)


@pytest.mark.parametrize("project", [_PROJECT, _INPUT_PROJECT], ids=["demo", "input"])
def test_production_and_tests_have_no_legacy_imports(project: Path) -> None:
    for path in project.rglob("*.py"):
        if any(part in {"build", "dist", ".venv"} for part in path.parts):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports = [node.module or ""]
            else:
                continue
            for module in imports:
                assert not module.startswith("integrations_v2."), (
                    f"{path}:{node.lineno}: app test imports an integration: {module}"
                )
                assert module.split(".")[0] not in _FORBIDDEN_ROOTS, (
                    f"{path}:{node.lineno}: legacy dependency {module}"
                )


@pytest.mark.parametrize("package", [_PACKAGE, _INPUT_PACKAGE], ids=["demo", "input"])
def test_package_has_no_legacy_paths_or_asset_links(package: Path) -> None:
    for path in package.rglob("*"):
        assert not path.is_symlink(), f"Package depends on a symlink: {path}"
        if path.suffix not in {
            ".py",
            ".sh",
            ".toml",
            ".json",
            ".kts",
            ".xml",
            ".cpp",
            ".h",
        }:
            continue
        text = path.read_text()
        for name in _FORBIDDEN_ROOTS[:3]:
            assert re.search(rf"\b{re.escape(name)}\b", text) is None, (
                f"{path}: legacy package/path {name}"
            )


@pytest.mark.parametrize("project", [_PROJECT, _INPUT_PROJECT], ids=["demo", "input"])
def test_package_dependency_closure_excludes_legacy_distributions(
    project: Path,
) -> None:
    config = tomllib.loads((project / "pyproject.toml").read_text())
    assert "cosmos-interactive-internal" not in config.get("tool", {}).get(
        "uv", {}
    ).get("sources", {})
    lock = tomllib.loads((_REPOSITORY / "uv.lock").read_text())
    packages: dict[str, list[dict]] = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], []).append(package)
    root = config["project"]["name"]
    assert packages[root][0]["source"]["editable"] == str(
        project.relative_to(_REPOSITORY)
    )
    declared = {Requirement(value).name for value in config["project"]["dependencies"]}
    assert {edge["name"] for edge in packages[root][0]["dependencies"]} == declared
    # Walk the resolved graph, including extras and every platform branch.
    # CPU CI need not install CloudXR or native GPU dependencies to enforce it.
    pending = [(root, frozenset())]
    seen = set()
    while pending:
        name, extras = key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        assert name not in {
            "cosmos-interactive-internal",
            "flashdreams-daydreams",
            "flashdreams-cosmos3-nano-sim-bimanual",
        }, name
        for package in packages[name]:
            assert not package["source"].get("editable", "").startswith("internal/"), (
                name
            )
            edges = list(package.get("dependencies", ()))
            for extra in extras:
                edges.extend(package.get("optional-dependencies", {}).get(extra, ()))
            pending.extend(
                (edge["name"], frozenset(edge.get("extra", ()))) for edge in edges
            )


def test_every_v2_module_loads_with_legacy_packages_blocked() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib
import importlib.abc
from pathlib import Path
import sys

sys.path[:0] = sys.argv[1:5]
blocked = tuple(sys.argv[5:])
class NoLegacyPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('V2 imported legacy code: ' + fullname)
sys.meta_path.insert(0, NoLegacyPackage())
for project, package in zip(sys.argv[1:3], ('handdreams', 'flashdreams_controller_input')):
    root = Path(project) / package
    for source in sorted(root.rglob('*.py')):
        relative = source.relative_to(root).with_suffix('')
        if relative.name == '__main__':
            continue
        parts = relative.parts[:-1] if relative.name == '__init__' else relative.parts
        importlib.import_module('.'.join((package, *parts)))
assert not any(name.split('.')[0] in blocked for name in sys.modules)
from apps.handdreams.tests.helpers import _application
app = _application()
app.init([])
assert app.scene_catalog == ("scene-a", "scene-b")
assert app.launch_config.isaac_teleop_python == Path.home() / ".local/share/flashdreams/isaac-teleop/bin/python"
app.close()
print('PASS: all V2 modules and application setup are independent')
""",
            str(_PACKAGE.parent),
            str(_INPUT_PACKAGE.parent),
            str(_REPOSITORY / "flashdreams"),
            str(_REPOSITORY),
            *_FORBIDDEN_ROOTS,
            "isaacteleop",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
