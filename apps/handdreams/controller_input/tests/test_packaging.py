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

"""Controller-plugin distribution metadata and standalone device assets."""

import configparser
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
def test_controller_wheel_owns_modes_and_device_assets(
    tmp_path: Path, from_sdist: bool
) -> None:
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    project = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(project / name, source / name)
    shutil.copytree(
        project / "flashdreams_controller_input",
        source / "flashdreams_controller_input",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        symlinks=True,
    )
    shutil.copytree(project / "isaac_teleop", source / "isaac_teleop")
    if from_sdist:
        _build(source, "sdist")
        with tarfile.open(next((source / "dist").glob("*.tar.gz"))) as archive:
            assert not any(member.issym() or member.islnk() for member in archive)
            archive.extractall(tmp_path / "sdist", filter="data")
        source = next((tmp_path / "sdist").iterdir())
    _build(source, "wheel")
    with zipfile.ZipFile(next((source / "dist").glob("*.whl"))) as archive:
        names = set(archive.namelist())
        package = "flashdreams_controller_input/"
        assert {
            package + "events.py",
            package + "quest_source.py",
            package + "isaac_teleop_source.py",
            package + "isaac_teleop_setup.py",
            package + "setup.py",
            package + "remote_quest.py",
            package + "quest_relay.py",
            package + "quest_webxr.py",
            package + "quest_usb_local.py",
            package + "quest-offhead-teleop.sh",
            package + "controller_server.py",
            package + "quest_remote_v2_window.py",
            package + "unified_source.py",
        } <= names
        for machine in ("aarch64", "x86_64"):
            lock = f"isaac_teleop/requirements-linux-{machine}-py312.txt"
            assert archive.read(lock) == (project / lock).read_bytes()
        assert not any(
            "quest_native" in name or name.endswith((".jar", ".so", ".dll"))
            for name in names
        )
        assert not any(
            name.startswith(("handdreams/", "cosmos_interactive_internal/"))
            for name in names
        )
        metadata = Parser().parsestr(
            archive.read(
                next(name for name in names if name.endswith("/METADATA"))
            ).decode()
        )
        requirements = {
            requirement.name: requirement
            for value in metadata.get_all("Requires-Dist", [])
            for requirement in (Requirement(value),)
        }
        assert set(requirements) == {
            "flashdreams",
            "numpy",
            "pytest",
            "websockets",
        }
        assert requirements["flashdreams"].extras == {"serving"}
        runtime = requirements["flashdreams"].marker
        assert runtime is not None
        assert runtime.evaluate({"extra": "runtime"})
        assert not runtime.evaluate({"extra": ""})
        assert not runtime.evaluate({"extra": "isaac"})
        assert {
            name
            for name, requirement in requirements.items()
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""})
        } == {"numpy"}
        assert {
            name
            for name, requirement in requirements.items()
            if requirement.marker is None
            or requirement.marker.evaluate({"extra": "webxr"})
        } == {"numpy", "websockets"}
        assert "isaac" not in metadata.get_all("Provides-Extra", [])
        entry_points = configparser.ConfigParser()
        entry_points.read_string(
            archive.read(
                next(name for name in names if name.endswith("/entry_points.txt"))
            ).decode()
        )
        assert dict(entry_points["flashdreams.client_windows_v2"]) == {
            "isaac-teleop-webrtc": (
                "flashdreams_controller_input.isaac_teleop_v2_window:ISAAC_TELEOP_WEBRTC_MODE"
            ),
            "quest-remote-webrtc": (
                "flashdreams_controller_input.quest_remote_v2_window:QUEST_REMOTE_WEBRTC_MODE"
            ),
            "quest-remote-native-window": (
                "flashdreams_controller_input.quest_remote_v2_window:QUEST_REMOTE_NATIVE_WINDOW_MODE"
            ),
        }
        assert dict(entry_points["flashdreams.setup"]) == {
            "isaac-teleop": "flashdreams_controller_input.isaac_teleop_setup:main",
        }
        assert dict(entry_points["console_scripts"]) == {
            "flashdreams-controller-input": "flashdreams_controller_input.setup:main",
            "flashdreams-quest-relay": "flashdreams_controller_input.quest_relay:main",
        }
        assert "flashdreams.applications_v2" not in entry_points
        installed = tmp_path / "installed"
        archive.extractall(installed)

    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib.abc
from importlib.metadata import entry_points
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

sys.path.insert(0, sys.argv[1])
class NoRuntimeOrSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'handdreams', 'cosmos_interactive_internal', 'isaacteleop', 'flashdreams', 'torch'}:
            raise AssertionError('controller tooling imported ' + fullname)
sys.meta_path.insert(0, NoRuntimeOrSDK())
from flashdreams_controller_input.quest_source import _launcher_script
root = Path(sys.argv[1])
launcher = _launcher_script()
assert launcher.is_relative_to(root) and launcher.is_file()
assert not any(path.is_symlink() for path in root.rglob('*'))
subprocess.run(['bash', '-n', str(launcher)], check=True)
from flashdreams_controller_input.quest_relay import main
sys.argv = ['flashdreams-quest-relay', '--help']
try:
    main()
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('remote relay did not handle --help')
from flashdreams_controller_input.isaac_teleop_setup import REQUIREMENTS
setup_entry, = entry_points(group='console_scripts', name='flashdreams-controller-input')
setup_main = setup_entry.load()
assert REQUIREMENTS == root / 'isaac_teleop'
assert len(list(REQUIREMENTS.glob('requirements-linux-*-py312.txt'))) == 2
try:
    setup_main(['setup', 'isaac-teleop', '--help'])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('isolated setup did not handle --help')
""",
            str(installed),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "--cloudxr-port" in probe.stdout
    for removed_option in ("--cloudxr-host", "--web-client-base", "--usb-local"):
        assert removed_option not in probe.stdout
    assert "--accept-eula" in probe.stdout
    assert "CLOUDXR_LICENSE" in probe.stdout


def _build(source: Path, artifact: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from setuptools.build_meta import build_{artifact}; build_{artifact}('dist')",
        ],
        cwd=source,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
