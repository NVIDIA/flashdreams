# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone CMake build-environment diagnostic regression check."""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import build_backend


if __name__ == "__main__":
    for platform in ("win32", "linux"):
        for hook in (
            "build_wheel",
            "build_editable",
            "build_sdist",
            "get_requires_for_build_wheel",
        ):
            with (
                patch.object(build_backend.sys, "platform", platform),
                patch.object(build_backend, "_setup_windows_env") as setup,
            ):
                assert getattr(build_backend, hook) is getattr(
                    build_backend.build, hook
                )
                if platform == "win32" and hook in {"build_wheel", "build_editable"}:
                    setup.assert_called_once_with()
                else:
                    setup.assert_not_called()
    with tempfile.TemporaryDirectory(prefix="partial VS ") as directory:
        (Path(directory) / "cl.exe").touch()
        partial = {"PATH": directory, "VSINSTALLDIR": directory}
        with (
            patch.dict(os.environ, partial, clear=True),
            patch.object(build_backend.shutil, "which", return_value=None),
            patch.object(build_backend, "set_meson_command"),
            patch.object(build_backend, "setup_vsenv") as setup,
        ):
            build_backend._setup_windows_env()
            setup.assert_called_once_with(force=True)
            assert "VSINSTALLDIR" not in os.environ
            assert directory not in os.environ["PATH"]
        with (
            patch.dict(os.environ, partial, clear=True),
            patch.object(build_backend.shutil, "which", return_value=None),
            patch.object(build_backend, "set_meson_command"),
            patch.object(
                build_backend, "setup_vsenv", side_effect=RuntimeError("no VS")
            ),
        ):
            try:
                build_backend._setup_windows_env()
            except RuntimeError as error:
                assert str(error) == "no VS"
            else:
                raise AssertionError("Missing Visual Studio must fail")
            assert dict(os.environ) == partial
    with (
        patch.dict(os.environ, {"INCLUDE": "headers", "LIB": "libraries"}),
        patch.object(build_backend.shutil, "which", return_value="available"),
        patch.object(build_backend, "setup_vsenv") as setup,
    ):
        build_backend._setup_windows_env()
        setup.assert_not_called()
    preflight = (
        Path(__file__).with_name("CMakeLists.txt").read_text().split("project(")[0]
    )
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "check.cmake"
        for windows, generator, missing in [
            (True, "Ninja", "WEBM_C_COMPILER"),
            (True, "Ninja", "CMAKE_RC_COMPILER"),
            (True, "Ninja", "CMAKE_MT"),
            (True, "Ninja", ""),
            (False, "Ninja", "CMAKE_MT"),
            (True, "Visual Studio 17 2022", "CMAKE_MT"),
        ]:
            script.write_text(
                f"set(CMAKE_HOST_WIN32 {'TRUE' if windows else 'FALSE'})\n"
                f'set(CMAKE_GENERATOR "{generator}")\n'
                "set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)\n"
                "set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)\n"
                + "".join(
                    f'set({name} "{"tool-NOTFOUND" if name == missing else "available"}")\n'
                    for name in ("WEBM_C_COMPILER", "CMAKE_RC_COMPILER", "CMAKE_MT")
                )
                + preflight
            )
            result = subprocess.run(
                ["cmake", "-P", str(script)], capture_output=True, text=True
            )
            should_fail = windows and generator == "Ninja" and bool(missing)
            assert (result.returncode != 0) == should_fail, result.stderr
            if should_fail:
                assert "Install Visual Studio C++ Build Tools" in " ".join(
                    result.stderr.split()
                ), result.stderr
    print("Build environment checks passed.")
