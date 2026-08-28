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

"""JIT compilation of the Vulkan torch extension.

This is a second, optional plugin alongside :mod:`._plugin`. It compiles
the Vulkan rasterizer (``ludus_renderer_vk_plugin``) which depends on
Vulkan headers and the Vulkan loader library at link time.

Importing this module is lazy and safe -- the actual compilation only
runs the first time :func:`_get_vk_plugin` is called. Failures surface
as ``RuntimeError`` from that call, which the :class:`LudusTimestampedContext`
constructor turns into a friendly :class:`ImportError` with installation
hints.
"""

from __future__ import annotations

import logging
import os
import shutil

import torch
import torch.utils.cpp_extension

_cached_plugin = None
_log = logging.getLogger("ludus_renderer.vk")
_dll_directory_handles = []


def _cuda_home() -> str:
    return (
        os.environ.get("CUDA_HOME")
        or os.environ.get("CUDA_PATH")
        or torch.utils.cpp_extension.CUDA_HOME
        or ""
    )


def _prepare_windows_build() -> None:
    """Make the MSVC linker and dependent runtime DLLs discoverable."""
    if os.name != "nt":
        return

    if shutil.which("cl.exe") is None:
        import glob

        pattern = (
            r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*"
            r"\bin\Hostx64\x64\cl.exe"
        )
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise RuntimeError(
                "Could not locate a supported Microsoft Visual C++ installation"
            )
        os.environ["PATH"] += os.pathsep + os.path.dirname(candidates[-1])

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    candidates = [os.path.join(os.path.dirname(torch.__file__), "lib")]
    cuda_home = _cuda_home()
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin"))
        # CUDA 13.2's Windows toolkit places nvJPEG in bin/x64 while the
        # import library remains under lib/x64.
        candidates.append(os.path.join(cuda_home, "bin", "x64"))
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        candidates.append(os.path.join(sdk, "Bin"))
    for path in candidates:
        if os.path.isdir(path):
            _dll_directory_handles.append(add_dll_directory(path))


def _resolve_vulkan_include() -> list[str]:
    """Find Vulkan headers. Try VULKAN_SDK, then common system paths."""
    candidates: list[str] = []
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for sub in ("Include", "include", os.path.join("x86_64", "include")):
            p = os.path.join(sdk, sub)
            if os.path.isdir(p):
                candidates.append(p)
    for p in ("/usr/include", "/usr/local/include"):
        if os.path.isdir(os.path.join(p, "vulkan")):
            candidates.append(p)
    return candidates


def _resolve_vulkan_libdir() -> list[str]:
    """Find the directory containing the platform Vulkan loader library."""
    dirs: list[str] = []
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for sub in ("Lib", "lib", os.path.join("x86_64", "lib")):
            p = os.path.join(sdk, sub)
            if os.path.isdir(p):
                dirs.append(p)
    for p in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib"):
        if any(
            f.startswith("libvulkan.so")
            for f in (os.listdir(p) if os.path.isdir(p) else [])
        ):
            dirs.append(p)
    return dirs


def _vulkan_available() -> tuple[bool, str]:
    """Check that Vulkan headers and loader library are present."""
    if not _resolve_vulkan_include():
        platform_hint = (
            "install the LunarG Vulkan SDK and set VULKAN_SDK"
            if os.name == "nt"
            else "install libvulkan-dev (Debian/Ubuntu) or set VULKAN_SDK"
        )
        return False, (f"Vulkan headers not found; {platform_hint} to its root.")
    libdirs = _resolve_vulkan_libdir()
    if os.name == "nt":
        if not any(
            os.path.isfile(os.path.join(path, "vulkan-1.lib")) for path in libdirs
        ):
            return False, (
                "Vulkan loader import library not found; install the LunarG "
                "Vulkan SDK and set VULKAN_SDK to its root."
            )
    elif not libdirs and not shutil.which("vulkaninfo"):
        return False, (
            "Vulkan loader (libvulkan.so) not found. Install "
            "libvulkan1 (Debian/Ubuntu) or the Vulkan SDK."
        )
    return True, ""


def _get_vk_plugin():
    """Compile (if needed) and return the Vulkan plugin module.

    Raises ``RuntimeError`` if Vulkan headers/loader are missing, or if
    JIT compilation fails.
    """
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    ok, msg = _vulkan_available()
    if not ok:
        raise RuntimeError(f"Vulkan backend unavailable: {msg}")

    _prepare_windows_build()

    common_opts = ["-DNVDR_TORCH", "-DFW_DO_NOT_OVERRIDE_NEW_DELETE"]
    cc_opts = common_opts + (["/wd4067", "/wd4624"] if os.name == "nt" else [])
    cuda_opts = common_opts + ["-lineinfo"]

    source_files = [
        "../_cpp/common/common.cpp",
        "../_cpp/common/vkutil.cpp",
        "../_cpp/render/ludus_timestamped_vk.cpp",
        "../_cpp/render/ludus_jpeg.cu",
        "../_cpp/bindings/torch_rasterize_vk.cpp",
    ]
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]

    extra_include_paths = _resolve_vulkan_include()
    vulkan_libdirs = _resolve_vulkan_libdir()
    if os.name == "nt":
        ldflags = ["cuda.lib", "vulkan-1.lib", "nvjpeg.lib"]
        libdirs = list(vulkan_libdirs)
        cuda_home = _cuda_home()
        if cuda_home:
            libdirs.append(os.path.join(cuda_home, "lib", "x64"))
        ldflags.extend(f"/LIBPATH:{path}" for path in dict.fromkeys(libdirs))
    else:
        ldflags = ["-lcuda", "-lvulkan", "-lnvjpeg"]
        for path in vulkan_libdirs:
            ldflags.insert(0, f"-L{path}")
            ldflags.insert(0, f"-Wl,-rpath,{path}")

    # Reset CUDA arch list to let PyTorch detect the installed GPU.
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    plugin_name = "ludus_renderer_vk_plugin"

    try:
        lock_fn = os.path.join(
            torch.utils.cpp_extension._get_build_directory(plugin_name, False), "lock"
        )
        if os.path.exists(lock_fn):
            _log.warning("Stale lock file in Vulkan plugin build dir: %s", lock_fn)
    except (OSError, RuntimeError) as exc:
        _log.debug("Could not inspect Vulkan plugin build directory: %s", exc)

    _log.info("Compiling Vulkan plugin (this may take a minute on first run)...")
    _cached_plugin = torch.utils.cpp_extension.load(
        name=plugin_name,
        sources=source_paths,
        extra_include_paths=extra_include_paths,
        extra_cflags=cc_opts,
        extra_cuda_cflags=cuda_opts,
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=True,
    )
    return _cached_plugin
