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

"""CPU checks for optional setup-provider dispatch."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from flashdreams_controller_input import setup

pytestmark = pytest.mark.ci_cpu


def test_dispatch_forwards_help_and_acceptance_without_loading_inference(monkeypatch):
    provider = Mock()
    lookup = Mock(return_value=[provider])
    monkeypatch.setattr(setup, "entry_points", lookup)
    setup.main(["setup", "isaac-teleop", "--accept-eula", "--python", "python3.12"])
    lookup.assert_called_once_with(group="flashdreams.setup", name="isaac-teleop")
    provider.load.return_value.assert_called_once_with(
        ["--accept-eula", "--python", "python3.12"]
    )
    provider.load.return_value.reset_mock()
    setup.main(["setup", "isaac-teleop", "--help"])
    provider.load.return_value.assert_called_once_with(["--help"])


@pytest.mark.parametrize("providers", [[], [Mock(), Mock()]])
def test_missing_or_ambiguous_setup_provider_is_actionable(
    providers, monkeypatch, capsys
):
    monkeypatch.setattr(setup, "entry_points", lambda **_: providers)
    with pytest.raises(SystemExit) as error:
        setup.main(["setup", "isaac-teleop"])
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert output.startswith("usage: flashdreams-controller-input ")
    assert "isaac-teleop" in output
    if not providers:
        assert "flashdreams-controller-input" in output


def test_setup_and_fresh_relay_process_share_default_across_working_directories(
    tmp_path,
):
    package = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "HOME": str(tmp_path / "home")}
    scripts = {
        "setup-from": """
import sys, venv
sys.path.insert(0, sys.argv[1])
from flashdreams_controller_input import isaac_teleop_setup
# Exercise CLI directory selection with a real empty venv, without SDK downloads.
def install(*, directory, python):
    venv.EnvBuilder(with_pip=False).create(directory)
isaac_teleop_setup.setup = install
isaac_teleop_setup.main(['--accept-eula'])
""",
        "launch-from": """
import argparse, subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from flashdreams_controller_input.isaac_teleop_source import IsaacTeleopSource, resolve_python
from flashdreams_controller_input.quest_webxr import add_relay_arguments
parser = argparse.ArgumentParser()
add_relay_arguments(parser)
args = parser.parse_args([])
expected = Path.home() / '.local/share/flashdreams/isaac-teleop'
python = resolve_python(args.isaac_teleop_python)
assert Path(python) == expected / 'bin/python'
assert IsaacTeleopSource().python_executable == python
result = subprocess.run([python, '-I', '-c', 'import sys; print(sys.prefix)'],
                        check=True, text=True, capture_output=True)
assert Path(result.stdout.strip()) == expected
""",
    }
    for name, script in scripts.items():
        directory = tmp_path / name
        directory.mkdir()
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(package)],
            cwd=directory,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not (directory / ".venv-isaac-teleop").exists()
