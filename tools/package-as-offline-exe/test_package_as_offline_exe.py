# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the offline packager command."""

import os
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from flashdreams.runtime_v2 import cli

pytestmark = pytest.mark.ci_cpu

_PACKAGER: dict[str, Any] = runpy.run_path(
    str(Path(__file__).with_name("package_as_offline_exe.py"))
)


@pytest.mark.parametrize(
    "config",
    (["--mode", "native-window"], ["--mode=webrtc"]),
)
def test_interactive_modes_can_be_packaged(config: list[str]) -> None:
    _PACKAGER["_validate_config"](config)


def test_noninteractive_mode_cannot_be_packaged() -> None:
    with pytest.raises(
        _PACKAGER["PackageError"],
        match="Packaging requires --mode native-window or webrtc",
    ):
        _PACKAGER["_validate_config"](["--mode", "mp4"])


def test_slangpy_shaders_are_bundled_beside_runtime_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packager_globals = _PACKAGER["_pyinstaller_command"].__globals__
    monkeypatch.setitem(
        packager_globals, "_module_search_path", lambda _module: tmp_path
    )
    monkeypatch.setitem(packager_globals, "_metadata_distributions", lambda _module: ())
    monkeypatch.setitem(
        packager_globals, "_local_dependency_modules", lambda _distributions: ()
    )
    monkeypatch.setitem(packager_globals, "_nvrtc_builtins", lambda: ())

    command = _PACKAGER["_pyinstaller_command"](
        launcher=tmp_path / "launcher.py",
        slug="example",
        application_module="example",
        build_root=tmp_path / "build",
    )

    add_data = command.index("--add-data")
    assert command[add_data + 1] == (
        f"{tmp_path / 'slangpy' / 'shaders'}{os.pathsep}shaders"
    )


def test_nvrtc_builtins_are_bundled_beside_nvrtc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packager_globals = _PACKAGER["_pyinstaller_command"].__globals__
    builtins = tmp_path / "nvidia" / "cu13" / "lib" / "libnvrtc-builtins.so.13.0"
    monkeypatch.setitem(
        packager_globals, "_module_search_path", lambda _module: tmp_path
    )
    monkeypatch.setitem(packager_globals, "_metadata_distributions", lambda _module: ())
    monkeypatch.setitem(
        packager_globals, "_local_dependency_modules", lambda _distributions: ()
    )
    monkeypatch.setitem(
        packager_globals,
        "_nvrtc_builtins",
        lambda: ((builtins, Path("nvidia/cu13/lib")),),
    )

    command = _PACKAGER["_pyinstaller_command"](
        launcher=tmp_path / "launcher.py",
        slug="example",
        application_module="example",
        build_root=tmp_path / "build",
    )

    binaries = [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--add-binary"
    ]
    assert f"{builtins}{os.pathsep}." in binaries
    assert f"{builtins}{os.pathsep}{Path('nvidia/cu13/lib')}" in binaries


def test_skip_runtime_validation_keeps_application_preload() -> None:
    parsed, command = _PACKAGER["_parse_arguments"](
        [
            "--skip-runtime-validation",
            ":::",
            "flashdreams-run-v2",
            "example",
            "--mode",
            "native-window",
        ]
    )

    assert parsed.skip_runtime_validation
    source = _PACKAGER["_preload_source"](
        "example", command[2:], skip_runtime_validation=True
    )
    assert "--preload-application" in source
    assert "--skip-preload-validation" in source


def test_runtime_validation_uses_no_window_mode() -> None:
    source = _PACKAGER["_preload_source"](
        "example",
        [
            "--mode",
            "native-window",
            "--pixel-width",
            "64",
            "--",
            "--model-option",
        ],
    )

    assert (
        "entrypoint(['example', '--preload-application', '--pixel-width', '64', "
        "'--', '--model-option'])" in source
    )


def test_launcher_keeps_runtime_and_application_overrides_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []
    monkeypatch.setattr(cli, "entrypoint", lambda arguments: launched.append(arguments))
    monkeypatch.setattr(
        sys,
        "argv",
        ["example", "--timeout", "5", "--", "--launch-option"],
    )
    for name in _PACKAGER["_CACHE_PATHS"]:
        monkeypatch.setenv(name, "")

    source = _PACKAGER["_launcher_source"](
        "example",
        ["--mode", "native-window", "--", "--embedded-option"],
    )
    launcher = tmp_path / "launcher.py"
    exec(compile(source, launcher, "exec"), {"__file__": str(launcher)})

    assert launched == [
        [
            "example",
            "--mode",
            "native-window",
            "--timeout",
            "5",
            "--",
            "--embedded-option",
            "--launch-option",
        ]
    ]


def test_preparation_issues_are_written_beside_installer_output(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "bundle"
    issues_path = staging / "PREPARATION_ISSUES.txt"

    environment = _PACKAGER["_cache_environment"](
        staging / "cache",
        preparation_issues_path=issues_path,
    )

    assert environment["FLASHDREAMS_PREPARATION_ISSUES_PATH"] == str(issues_path)
    assert issues_path.parent == staging
