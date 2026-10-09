# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the offline packager command."""

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from torch.utils.cpp_extension import verify_ninja_availability

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


def test_required_runtime_assets_are_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imgui_fonts = tmp_path / "imgui_bundle/assets/fonts"
    imgui_fonts.mkdir(parents=True)
    (imgui_fonts / "DroidSans.ttf").write_bytes(b"font")
    (imgui_fonts / "Inconsolata-Medium.ttf").write_bytes(b"second font")
    packager_globals = _PACKAGER["_pyinstaller_command"].__globals__
    monkeypatch.setitem(
        packager_globals, "_module_search_path", lambda _module: tmp_path
    )
    monkeypatch.setitem(packager_globals, "_metadata_distributions", lambda _module: ())
    monkeypatch.setitem(
        packager_globals, "_local_dependency_modules", lambda _distributions: ()
    )
    monkeypatch.setitem(packager_globals, "_nvrtc_builtins", lambda: ())
    ninja = _PACKAGER["_ninja_executable"]()

    command = _PACKAGER["_pyinstaller_command"](
        launcher=tmp_path / "launcher.py",
        slug="example",
        application_module="example",
        build_root=tmp_path / "build",
    )

    assert "--onefile" in command
    assert "--onedir" not in command
    assert "--contents-directory" not in command
    bundled_data = [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--add-data"
    ]
    assert f"{tmp_path / 'slangpy' / 'shaders'}{os.pathsep}shaders" in bundled_data
    assert f"{imgui_fonts}{os.pathsep}imgui_bundle/assets/fonts" in bundled_data
    assert f"{ninja}{os.pathsep}." in [
        command[index + 1]
        for index, argument in enumerate(command)
        if argument == "--add-binary"
    ]
    runtime_root = tmp_path / "data"
    runtime_root.mkdir()
    shutil.copy2(ninja, runtime_root / ninja.name)
    monkeypatch.setenv("PATH", str(runtime_root))
    verify_ninja_availability()


def test_nvrtc_builtins_are_bundled_beside_nvrtc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imgui_fonts = tmp_path / "imgui_bundle/assets/fonts"
    imgui_fonts.mkdir(parents=True)
    (imgui_fonts / "DroidSans.ttf").write_bytes(b"font")
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
    monkeypatch.setitem(
        packager_globals, "_ninja_executable", lambda: tmp_path / "ninja.exe"
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


@pytest.mark.parametrize("custom_unpack_dir", [False, True])
def test_frozen_launcher_runs_from_bundle_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_unpack_dir: bool
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    launcher = bundle / "launcher.py"
    launcher.write_text(
        _PACKAGER["_launcher_source"](
            "example", ["--mode", "native-window"], "test-seed"
        ),
        encoding="utf-8",
    )
    physx_source = bundle / "cache" / "ludus" / "physx" / "source"
    physx_source.mkdir(parents=True)
    for index in range(5):
        (physx_source / f"file-{index}.txt").write_text("prepared", encoding="utf-8")
    long_physx_name = "physx-" + "x" * 170 + ".txt"
    long_physx_file = physx_source / long_physx_name
    hub_relative = (
        Path("datasets--nvidia--omni-dreams-samples")
        / "snapshots"
        / ("c" * 40)
        / "data"
        / "single_view"
        / "239560dc-33d1-11ef-9720-00044bcbccac"
        / ("239560dc-33d1-11ef-9720-00044bcbccac_" + "1" * 70 + ".mp4")
    )
    long_hf_file = bundle / "cache" / "huggingface" / "hub" / hub_relative
    for path in (long_physx_file, long_hf_file):
        filesystem_path = Path("\\\\?\\" + str(path)) if os.name == "nt" else path
        filesystem_path.parent.mkdir(parents=True, exist_ok=True)
        filesystem_path.write_text("prepared", encoding="utf-8")
    assert len(str(long_hf_file)) > 260
    monkeypatch.setitem(
        _PACKAGER["_archive_cache_to_file_limit"].__globals__,
        "_ONEFILE_MAX_FILES",
        3,
    )
    _PACKAGER["_archive_cache_to_file_limit"](bundle)
    assert (bundle / "cache-extras.zip").is_file()
    assert not (bundle / "cache" / "ludus").exists()
    assert sum(path.is_file() for path in bundle.rglob("*")) <= 3
    stubs = tmp_path / "stubs"
    for package in ("flashdreams", "flashdreams/runtime_v2", "torch", "torch/utils"):
        directory = stubs / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").touch()
    (stubs / "torch/utils/cpp_extension.py").write_text(
        "def load(*args, **kwargs): pass\n"
    )
    (stubs / "flashdreams/runtime_v2/cli.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "def split_arguments(args): return args, []\n"
        "def entrypoint(args):\n"
        "    assert Path.cwd() == Path(os.environ['EXPECTED_BUNDLE_DIR'])\n"
        "    assert '--unpack-dir' not in args\n"
        "    assert os.environ['HF_HUB_CACHE'].startswith(os.environ['EXPECTED_LONG_PATH_PREFIX'])\n"
        "    assert Path(os.environ['LUDUS_PHYSX_CACHE']).samefile(Path(os.environ['EXPECTED_CACHE_DIR']) / 'ludus' / 'physx')\n"
        "    assert (Path(os.environ['LUDUS_PHYSX_CACHE']) / 'source' / 'file-0.txt').read_text() == 'prepared'\n"
        "    assert (Path(os.environ['LUDUS_PHYSX_CACHE']) / 'source' / os.environ['EXPECTED_LONG_PHYSX']).read_text() == 'prepared'\n"
        "    assert (Path(os.environ['HF_HUB_CACHE']) / os.environ['EXPECTED_LONG_HF']).read_text() == 'prepared'\n"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    profile = tmp_path / "profile"
    profile.mkdir()
    unpack_root = (
        tmp_path / "custom-unpack"
        if custom_unpack_dir
        else profile / "flashdreams" / "unpacked"
    )
    cache_root = unpack_root / "example" / "cache"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(stubs),
        EXPECTED_BUNDLE_DIR=str(bundle),
        EXPECTED_CACHE_DIR=str(cache_root),
        EXPECTED_LONG_PATH_PREFIX="\\\\?\\" if os.name == "nt" else "",
        EXPECTED_LONG_PHYSX=long_physx_name,
        EXPECTED_LONG_HF=str(hub_relative),
        USERPROFILE=str(profile),
        HOME=str(profile),
    )
    env.pop("FLASHDREAMS_RUNTIME_CACHE_DIR", None)
    if custom_unpack_dir:
        env["FLASHDREAMS_RUNTIME_CACHE_DIR"] = str(tmp_path / "env-cache")
    bootstrap = (
        "import runpy, sys\n"
        "sys.frozen = True\n"
        f"sys.executable = {str(bundle / 'example.exe')!r}\n"
        f"sys._MEIPASS = {str(bundle / 'data')!r}\n"
        f"sys.argv = {['example.exe', '--unpack-dir', str(unpack_root)] if custom_unpack_dir else ['example.exe']!r}\n"
        f"runpy.run_path({str(launcher)!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", bootstrap],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_preparation_issues_are_written_beside_installer_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "bundle"
    issues_path = staging / "PREPARATION_ISSUES.txt"
    monkeypatch.setenv("PATH", "")

    environment = _PACKAGER["_cache_environment"](
        staging / "cache",
        preparation_issues_path=issues_path,
    )

    assert environment["FLASHDREAMS_PREPARATION_ISSUES_PATH"] == str(issues_path)
    assert issues_path.parent == staging
    assert shutil.which("ninja", path=environment["PATH"]) is not None


def test_preload_failure_reports_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packager_globals = _PACKAGER["package"].__globals__

    def failed_preload(
        args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        kwargs["stdout"].write(
            b"Traceback (most recent call last):\nValueError: preload failed\n"
        )
        return subprocess.CompletedProcess(args, 1)

    monkeypatch.setattr(packager_globals["subprocess"], "run", failed_preload)
    with pytest.raises(_PACKAGER["PackageError"], match="ValueError: preload failed"):
        _PACKAGER["package"](
            "example", ["--mode", "native-window"], tmp_path / "bundle"
        )


def test_completed_bundle_survives_output_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packager_globals = _PACKAGER["package"].__globals__
    destination = tmp_path / "bundle"

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args[1].endswith("preload.py"):
            return subprocess.CompletedProcess(args, 0)
        build = kwargs["cwd"] / ".build" / "dist"
        build.mkdir(parents=True)
        (build / _PACKAGER["_executable_name"]("example")).touch()
        destination.mkdir()
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(packager_globals["subprocess"], "run", fake_run)
    monkeypatch.setitem(
        packager_globals, "_pyinstaller_command", lambda **_: ["python", "pyinstaller"]
    )
    with pytest.raises(_PACKAGER["PackageError"], match="retained at"):
        _PACKAGER["package"]("example", ["--mode", "native-window"], destination)

    staging = next(tmp_path.glob(".fd-*.tmp"))
    assert (staging / _PACKAGER["_executable_name"]("example")).is_file()
    assert not (staging / "data").exists()
