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

"""CPU checks for sidecar consent, validation, and atomic publication."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from flashdreams_controller_input import isaac_teleop_setup as setup

pytestmark = pytest.mark.ci_cpu


def test_web_client_preparation_reuses_complete_cache_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(tmp_path))
    for name in ("index.html", "bundle.js", "bundle.emulator.js"):
        (tmp_path / name).write_text("cached asset")
    monkeypatch.setattr(
        setup, "_run", lambda *_, **__: pytest.fail("No subprocess for cached assets")
    )
    assert setup.prepare_web_client(python="unused") == tmp_path


@pytest.mark.parametrize("missing", ["index.html", "bundle.js", "bundle.emulator.js"])
@pytest.mark.parametrize("empty", [False, True])
def test_web_client_preparation_uses_isolated_sdk_and_validates_cache(
    tmp_path, monkeypatch, missing, empty
):
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(tmp_path))
    for name in ("index.html", "bundle.js", "bundle.emulator.js"):
        if name != missing:
            (tmp_path / name).write_text("cached asset")
        elif empty:
            (tmp_path / name).touch()
    calls = []

    def fetch(command, *, timeout):
        calls.append(command)
        assert command[:3] == ["/isolated/bin/python", "-I", "-c"]
        assert "require_web_client_static_dir()" in command[3]
        assert (
            "https://nvidia.github.io/IsaacCapture/client/release-1.4.x/" in command[3]
        )
        assert timeout == 400.0
        (tmp_path / missing).write_text("downloaded asset")
        return ""

    monkeypatch.setattr(setup, "_run", fetch)
    assert setup.prepare_web_client(python=" /isolated/bin/python ") == tmp_path
    assert len(calls) == 1
    assert setup.prepare_web_client(python="/isolated/bin/python") == tmp_path
    assert len(calls) == 1
    for name in ("index.html", "bundle.js", "bundle.emulator.js"):
        assert (tmp_path / name).read_text() == (
            "downloaded asset" if name == missing else "cached asset"
        )


@pytest.mark.parametrize(
    "failure", [None, OSError("offline"), ValueError("SDK failed")]
)
def test_web_client_preparation_fails_on_sdk_error_or_missing_emulator(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(tmp_path))
    for name in ("index.html", "bundle.js"):
        (tmp_path / name).write_text("cached asset")

    def fetch(*_, **__):
        if failure is not None:
            raise failure
        return ""

    monkeypatch.setattr(setup, "_run", fetch)
    with pytest.raises(
        ValueError, match="Could not prepare.*TELEOP_WEB_CLIENT_STATIC_DIR"
    ):
        setup.prepare_web_client(python="/isolated/bin/python")
    for name in ("index.html", "bundle.js"):
        assert (tmp_path / name).read_text() == "cached asset"


def test_setup_refuses_noninteractive_implicit_acceptance(monkeypatch, capsys):
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        setup, "setup", lambda **_: pytest.fail("No mutation before consent")
    )
    with pytest.raises(SystemExit) as error:
        setup.main([])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert setup.ISAAC_TELEOP_REQUIREMENT in output.out
    assert setup.CLOUDXR_EULA_URL in output.out
    assert "--accept-eula" in output.err


@pytest.mark.parametrize("answer", ["", "y", "no"])
def test_setup_decline_is_side_effect_free(answer, monkeypatch):
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: answer)
    monkeypatch.setattr(
        setup, "setup", lambda **_: pytest.fail("No mutation on decline")
    )
    with pytest.raises(SystemExit) as error:
        setup.main([])
    assert error.value.code == 1


@pytest.mark.parametrize("arguments", [[], ["--accept-eula"]])
def test_setup_explicit_consent_dispatches(arguments, monkeypatch, capsys):
    monkeypatch.setattr(setup.sys.stdin, "isatty", lambda: True)

    def accept(_):
        assert setup.CLOUDXR_EULA_URL in capsys.readouterr().out
        return "yes"

    monkeypatch.setattr("builtins.input", accept)
    with patch.object(setup, "setup") as install:
        setup.main(arguments)
    install.assert_called_once_with(
        directory=setup.DEFAULT_ISAAC_TELEOP_DIRECTORY, python="python3.12"
    )


@pytest.mark.parametrize(
    "system,machine", [("Darwin", "arm64"), ("Windows", "AMD64"), ("Linux", "ppc64le")]
)
def test_unsupported_host_fails_before_subprocess(system, machine, monkeypatch):
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        setup, "_run", lambda _: pytest.fail("No process for unsupported host")
    )
    with pytest.raises(ValueError, match="requires Linux"):
        setup._platform("python3.12")


@pytest.mark.parametrize(
    "version,machine,libc",
    [
        ([3, 11], "x86_64", ["glibc", "2.35"]),
        ([3, 12], "aarch64", ["glibc", "2.35"]),
        ([3, 12], "x86_64", ["glibc", "2.34"]),
        ([3, 12], "x86_64", ["musl", "1.2"]),
    ],
)
def test_incompatible_interpreter_fails(version, machine, libc, monkeypatch):
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        setup,
        "_run",
        lambda _: json.dumps([version, "Linux", machine, "CPython", libc]),
    )
    with pytest.raises(ValueError):
        setup._platform("python")


@pytest.mark.parametrize("machine", ["x86_64", "aarch64"])
def test_setup_publishes_validated_env_and_reuses_without_install(
    machine, tmp_path, monkeypatch
):
    destination = tmp_path / ".venv-isaac-teleop"
    monkeypatch.setattr(setup, "_platform", lambda _: machine)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return ""

    def validate(directory, requirements):
        calls.append(["validate", directory, requirements.name])
        if directory != destination:
            assert not destination.exists()
            (directory / "validated").write_text("ready")
        assert requirements.name == f"requirements-linux-{machine}-py312.txt"

    monkeypatch.setattr(setup, "_run", run)
    monkeypatch.setattr(setup, "_validate", validate)
    setup.setup(directory=destination, python="python3.12")
    assert destination.is_symlink()
    backing = destination.resolve()
    assert (destination / "validated").read_text() == "ready"
    assert calls[0][:4] == ["python3.12", "-I", "-m", "venv"]
    install = calls[1]
    assert {
        "--require-hashes",
        "--only-binary=:all:",
        "https://pypi.org/simple",
    } <= set(install)
    assert "--upgrade" not in install
    assert "--accept-eula" not in install
    assert (
        json.loads((destination / setup.MARKER).read_text())["package"]
        == setup.ISAAC_TELEOP_REQUIREMENT
    )
    calls.clear()
    setup.setup(directory=destination, python="python3.12")
    assert destination.resolve() == backing
    assert calls == [
        ["validate", destination, f"requirements-linux-{machine}-py312.txt"]
    ]


@pytest.mark.parametrize("failure_step", ["venv", "install", "validate"])
def test_failed_setup_cleans_staging_without_publishing(
    failure_step, tmp_path, monkeypatch
):
    monkeypatch.setattr(setup, "_platform", lambda _: "x86_64")

    def run(command, **kwargs):
        if failure_step in command:
            raise ValueError("simulated failure")
        return ""

    def validate(*_):
        if failure_step == "validate":
            raise ValueError("doctor failure")

    monkeypatch.setattr(setup, "_run", run)
    monkeypatch.setattr(setup, "_validate", validate)
    with pytest.raises(ValueError, match="No existing environment was replaced"):
        setup.setup(directory=tmp_path / ".venv-isaac-teleop", python="python")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["directory", "file", "broken-symlink"])
def test_existing_unmanaged_environment_is_preserved(kind, tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "_platform", lambda _: "x86_64")
    monkeypatch.setattr(
        setup, "_run", lambda _: pytest.fail("No mutation of existing env")
    )
    destination = tmp_path / ".venv-isaac-teleop"
    if kind == "directory":
        destination.mkdir()
        (destination / "keep").write_text("existing")
    elif kind == "file":
        destination.write_text("existing")
    else:
        destination.symlink_to("missing")
    before = destination.lstat()
    with pytest.raises(ValueError, match="was preserved"):
        setup.setup(directory=destination, python="python")
    after = destination.lstat()
    assert (after.st_ino, after.st_mode, after.st_mtime_ns) == (
        before.st_ino,
        before.st_mode,
        before.st_mtime_ns,
    )
    if kind == "directory":
        assert (destination / "keep").read_text() == "existing"
    elif kind == "file":
        assert destination.read_text() == "existing"
    else:
        assert destination.readlink() == Path("missing")


def test_concurrent_environment_wins_without_replacement(tmp_path, monkeypatch):
    destination = tmp_path / ".venv-isaac-teleop"
    monkeypatch.setattr(setup, "_platform", lambda _: "x86_64")
    monkeypatch.setattr(setup, "_run", lambda _, **kwargs: "")

    def validate(*_):
        destination.mkdir()
        (destination / "keep").write_text("other setup")

    monkeypatch.setattr(setup, "_validate", validate)
    with pytest.raises(ValueError, match="No existing environment was replaced"):
        setup.setup(directory=destination, python="python")
    assert (destination / "keep").read_text() == "other setup"
    assert list(tmp_path.iterdir()) == [destination]


def test_matching_record_with_damaged_packages_is_not_replaced(tmp_path, monkeypatch):
    destination = tmp_path / ".venv-isaac-teleop"
    monkeypatch.setattr(setup, "_platform", lambda _: "x86_64")
    monkeypatch.setattr(setup, "_run", lambda _, **kwargs: "")
    monkeypatch.setattr(setup, "_validate", lambda *_: None)
    setup.setup(directory=destination, python="python")
    before = destination.resolve()

    def damaged(*_):
        raise ValueError("Installed packages do not match")

    monkeypatch.setattr(setup, "_validate", damaged)
    with pytest.raises(ValueError, match="was preserved"):
        setup.setup(directory=destination, python="python")
    assert destination.resolve() == before


def test_validation_checks_closure_pip_and_existing_doctor(tmp_path, monkeypatch):
    requirements = setup.REQUIREMENTS / "requirements-linux-x86_64-py312.txt"
    versions = setup._versions(requirements)
    assert len(versions) == 8
    installed = [
        {"name": name, "version": version} for name, version in versions.items()
    ]
    installed.append({"name": "pip", "version": "26.0"})
    monkeypatch.setattr(setup, "_platform", lambda _: "x86_64")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return json.dumps(installed) if "list" in command else "doctor ok\n"

    monkeypatch.setattr(setup, "_run", run)
    setup._validate(tmp_path, requirements)
    assert calls[1][-1] == "check"
    assert calls[2][-1] == "doctor"
    assert calls[2][-2].endswith("isaac_teleop_bridge.py")
    assert all("--accept-eula" not in command for command in calls)
    installed[0]["version"] = "0.0"
    with pytest.raises(ValueError, match="do not match"):
        setup._validate(tmp_path, requirements)


def test_subprocess_isolation_and_actionable_output(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/main/environment")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://unreviewed.invalid")
    with patch.object(
        setup.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, stdout="ok"),
    ) as run:
        assert setup._run(["python", "doctor"]) == "ok"
    environment = run.call_args.kwargs["env"]
    assert "PYTHONPATH" not in environment and "PIP_EXTRA_INDEX_URL" not in environment
    assert environment["PIP_CONFIG_FILE"] == setup.os.devnull
    with patch.object(
        setup.subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(
            1, ["doctor"], output="stdout", stderr="missing library"
        ),
    ):
        with pytest.raises(ValueError, match="missing library"):
            setup._run(["doctor"])


@pytest.mark.parametrize("failure", [None, "hash", "doctor"])
def test_real_offline_install_preserves_venv_scripts_and_rejects_failures(
    failure,
    tmp_path,
    monkeypatch,
):
    python = os.environ.get("FLASHDREAMS_SETUP_TEST_PYTHON", sys.executable)
    probe = subprocess.run(
        [
            python,
            "-I",
            "-c",
            "import ensurepip; import sys; assert sys.version_info[:2] == (3, 12)",
        ],
        capture_output=True,
    )
    if (
        probe.returncode
        or platform.system() != "Linux"
        or platform.machine() not in {"x86_64", "aarch64"}
    ):
        pytest.skip("Real installation requires Linux CPython 3.12 with ensurepip")
    try:
        setup._platform(python)
    except ValueError as error:
        pytest.skip(str(error))

    # A first-party fixture wheel tests real pip without downloading any SDK.
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / "flashdreams_setup_probe-0.0.0-py3-none-any.whl"
    info = "flashdreams_setup_probe-0.0.0.dist-info/"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "flashdreams_setup_probe.py",
            'def main(): print("fixture console script works")\n',
        )
        archive.writestr(
            info + "METADATA",
            "Metadata-Version: 2.1\nName: flashdreams-setup-probe\nVersion: 0.0.0\n",
        )
        archive.writestr(
            info + "WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            info + "entry_points.txt",
            "[console_scripts]\nflashdreams-setup-probe = flashdreams_setup_probe:main\n",
        )
        archive.writestr(
            info + "RECORD",
            "".join(f"{name},,\n" for name in archive.namelist()) + f"{info}RECORD,,\n",
        )
    digest = (
        "0" * 64
        if failure == "hash"
        else hashlib.sha256(wheel.read_bytes()).hexdigest()
    )
    requirements = tmp_path / f"requirements-linux-{platform.machine()}-py312.txt"
    requirements.write_text(f"flashdreams-setup-probe==0.0.0 --hash=sha256:{digest}\n")
    doctor = tmp_path / "isaac_teleop_bridge.py"
    doctor.write_text(
        "import importlib.metadata,sys\n"
        "assert sys.argv[1:] == ['doctor']\n"
        "assert importlib.metadata.version('flashdreams-setup-probe') == '0.0.0'\n"
        + (
            "raise RuntimeError('fixture doctor failed')\n"
            if failure == "doctor"
            else "print('fixture doctor passed')\n"
        )
    )
    monkeypatch.setattr(setup, "REQUIREMENTS", tmp_path)
    monkeypatch.setattr(setup, "__file__", str(tmp_path / "isaac_teleop_setup.py"))
    destination = tmp_path / "published sidecar"
    original_run = setup._run
    installs = []

    def offline_run(command, **kwargs):
        if "install" in command:
            assert not destination.exists()
            assert "--require-hashes" in command and "--only-binary=:all:" in command
            command = [*command, "--no-index", "--find-links", str(wheels)]
            installs.append(command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(setup, "_run", offline_run)
    monkeypatch.setenv("PIP_TARGET", str(tmp_path / "must-not-write"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "invalid-python-home"))
    before = set(tmp_path.iterdir())
    if failure:
        with pytest.raises(
            ValueError, match="No existing environment was replaced"
        ) as error:
            setup.setup(directory=destination, python=python)
        assert ("HASHES" if failure == "hash" else "fixture doctor failed") in str(
            error.value
        )
        assert set(tmp_path.iterdir()) == before
        return

    setup.setup(directory=destination, python=python)
    backing = destination.resolve()
    assert destination.is_symlink() and backing != destination
    assert not (tmp_path / "must-not-write").exists()
    # Renaming a venv would leave this script's absolute shebang broken.
    assert (
        original_run([str(destination / "bin/flashdreams-setup-probe")]).strip()
        == "fixture console script works"
    )
    setup.setup(directory=destination, python="missing-bootstrap-python")
    assert len(installs) == 1 and destination.resolve() == backing
    doctor.write_text("raise RuntimeError('fixture doctor failed after publication')\n")
    with pytest.raises(ValueError, match="was preserved"):
        setup.setup(directory=destination, python=python)
    assert destination.resolve() == backing
    assert (
        original_run([str(destination / "bin/flashdreams-setup-probe")]).strip()
        == "fixture console script works"
    )


def test_interrupt_after_publication_preserves_completed_environment(
    tmp_path, monkeypatch
):
    destination = tmp_path / ".venv-isaac-teleop"
    monkeypatch.setattr(setup, "_platform", lambda _: "x86_64")
    monkeypatch.setattr(setup, "_run", lambda _, **kwargs: "")
    monkeypatch.setattr(setup, "_validate", lambda *_: None)
    symlink = Path.symlink_to

    def publish_then_interrupt(path, target, **kwargs):
        symlink(path, target, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "symlink_to", publish_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        setup.setup(directory=destination, python="python")
    assert destination.is_symlink() and destination.is_dir()
    assert (destination / setup.MARKER).is_file()
    setup.setup(directory=destination, python="missing-bootstrap-python")


def test_interrupt_before_publication_cleans_staging_and_reports_recovery(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(setup, "_platform", lambda _: "x86_64")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(setup, "_run", interrupt)
    with pytest.raises(SystemExit) as error:
        setup.main(
            ["--accept-eula", "--directory", str(tmp_path / ".venv-isaac-teleop")]
        )
    assert error.value.code == 130
    assert not list(tmp_path.iterdir())
    assert (
        "rerun flashdreams-controller-input setup isaac-teleop"
        in capsys.readouterr().err
    )


def test_stalled_subprocess_has_a_deadline():
    with pytest.raises(ValueError, match="Command timed out after"):
        setup._run(
            [sys.executable, "-I", "-c", "import time; time.sleep(10)"], timeout=0.05
        )
