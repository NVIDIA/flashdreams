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

"""Standalone V2 wheels without scenario datasets or model dependencies."""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("from_sdist", [False, True])
def test_built_v2_wheel_excludes_scenario_data_and_device_assets(
    tmp_path: Path, from_sdist: bool
) -> None:
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    project = Path(__file__).resolve().parents[1]
    source = tmp_path / "apps" / "handdreams"
    source.mkdir(parents=True)
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(project / name, source / name)
    shutil.copytree(
        project / "handdreams",
        source / "handdreams",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        symlinks=True,
    )
    if from_sdist:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from setuptools.build_meta import build_sdist; build_sdist('dist')",
            ],
            cwd=source,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with tarfile.open(next((source / "dist").glob("*.tar.gz"))) as archive:
            assert not any("/bundled_scenes/" in member.name for member in archive)
            assert not any(member.name.endswith(".pkl") for member in archive)
            assert {
                member.name.split("/handdreams/assets/tutorials/", 1)[1]
                for member in archive
                if "/handdreams/assets/tutorials/" in member.name and member.isfile()
            } == {"press_triggers.mp4", "align_arms.mp4"}
            unpacked = tmp_path / "sdist"
            archive.extractall(unpacked, filter="data")
        source = next(unpacked.iterdir())

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('dist')",
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next((source / "dist").glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert not any("/bundled_scenes/" in name for name in names)
        assert not any(
            name.endswith(("/scene.json", "/first_frame.jpg")) for name in names
        )
        assert not any(name.endswith(".pkl") for name in names)
        assert {
            "handdreams/assets/tutorials/press_triggers.mp4",
            "handdreams/assets/tutorials/align_arms.mp4",
        } <= names
        assert "handdreams/controllers/quest_source.py" not in names
        assert "handdreams/controllers/isaac_teleop_source.py" not in names
        assert "handdreams/controllers/quest-offhead-teleop.sh" not in names
        assert not any(name.startswith("handdreams/normalizers/") for name in names)
        assert not any(name.startswith("handdreams/model/") for name in names)
        assert not any(name.startswith("handdreams/quest_native/") for name in names)
        assert not any(
            name.startswith("cosmos_interactive_internal/") for name in names
        )
        metadata = Parser().parsestr(
            archive.read(
                next(name for name in names if name.endswith("/METADATA"))
            ).decode()
        )
        requirements = {
            Requirement(requirement).name
            for requirement in metadata.get_all("Requires-Dist", [])
        }
        assert "flashdreams-controller-input[runtime,webxr]" in metadata.get_all(
            "Requires-Dist", []
        )
        assert "flashdreams-controller-input" in requirements
        assert "isaacteleop" not in requirements
        assert "flashdreams-cosmos3-nano-sim-bimanual" not in requirements
        assert "cosmos-interactive-internal" not in requirements
        assert not any(name.endswith("/entry_points.txt") for name in names)
        archive.extractall(installed)

    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
from pathlib import Path
import importlib.abc
import sys

sys.path[:0] = sys.argv[1:4]
class AppBoundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'internal', 'cosmos_interactive_internal', 'cosmos3_nano_sim_bimanual'}:
            raise AssertionError(f'application imported model package: {fullname}')
sys.meta_path.insert(0, AppBoundary())
from handdreams import app, scenes

root = Path(sys.argv[1])
assert Path(scenes.__file__).is_relative_to(root)
assert not any(path.is_symlink() for path in root.rglob('*'))
assert not list(root.rglob('scene.json'))
assert not list(root.rglob('first_frame.jpg'))
print('PASS: app wheel imports without bundled scenarios or a model')
""",
            str(installed),
            str(project / "controller_input"),
            str(project.parents[1] / "flashdreams"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
