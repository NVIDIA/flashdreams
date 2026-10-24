# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scikit-build backend with automatic Windows compiler initialization."""

import os
import shutil
import sys
from pathlib import Path

# These dependencies are installed in the isolated build environment.
from mesonbuild import mesonmain  # ty: ignore[unresolved-import]
from mesonbuild.utils.universal import set_meson_command  # ty: ignore[unresolved-import]
from mesonbuild.utils.vsenv import setup_vsenv  # ty: ignore[unresolved-import]
from scikit_build_core import build  # ty: ignore[unresolved-import]


def _setup_windows_env():
    """Activate MSVC even when PATH contains only part of its toolchain."""
    if all(shutil.which(tool) for tool in ("cl", "rc", "mt")) and all(
        os.environ.get(key) for key in ("INCLUDE", "LIB")
    ):
        return
    previous = os.environ.copy()
    try:
        # Meson assumes cl.exe or VSINSTALLDIR alone means activation is complete.
        os.environ.pop("VSINSTALLDIR", None)
        os.environ["PATH"] = os.pathsep.join(
            directory
            for directory in os.environ.get("PATH", "").split(os.pathsep)
            if not (Path(directory.strip('"')) / "cl.exe").is_file()
        )
        set_meson_command(mesonmain.__file__)
        setup_vsenv(force=True)
    except Exception:
        os.environ.clear()
        os.environ.update(previous)
        raise


def __getattr__(name: str):
    """Initialize the compiler environment before delegating native build hooks."""
    hook = getattr(build, name)
    if sys.platform == "win32" and name in {"build_wheel", "build_editable"}:
        _setup_windows_env()
    return hook
