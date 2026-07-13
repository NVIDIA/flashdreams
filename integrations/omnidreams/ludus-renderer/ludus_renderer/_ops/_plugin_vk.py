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


def _resolve_vulkan_include() -> list[str]:
    """Find Vulkan headers. Try VULKAN_SDK, then common system paths."""
    candidates: list[str] = []
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for sub in ("include", os.path.join("x86_64", "include")):
            p = os.path.join(sdk, sub)
            if os.path.isdir(p):
                candidates.append(p)
    for p in ("/usr/include", "/usr/local/include"):
        if os.path.isdir(os.path.join(p, "vulkan")):
            candidates.append(p)
    return candidates


def _resolve_vulkan_libdir() -> list[str]:
    """Find the directory containing libvulkan.so."""
    dirs: list[str] = []
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for sub in ("lib", os.path.join("x86_64", "lib")):
            p = os.path.join(sdk, sub)
            if os.path.isdir(p):
                dirs.append(p)
    for p in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib"):
        if any(f.startswith("libvulkan.so") for f in (os.listdir(p) if os.path.isdir(p) else [])):
            dirs.append(p)
    return dirs


def _vulkan_available() -> tuple[bool, str]:
    """Check that Vulkan headers and loader library are present."""
    if os.name == "nt":
        return False, "Vulkan backend is currently only supported on Linux."

    if not _resolve_vulkan_include():
        return False, ("Vulkan headers not found. Install libvulkan-dev "
                       "(Debian/Ubuntu) or set VULKAN_SDK to the Vulkan SDK root.")
    if not _resolve_vulkan_libdir() and not shutil.which("vulkaninfo"):
        return False, ("Vulkan loader (libvulkan.so) not found. Install "
                       "libvulkan1 (Debian/Ubuntu) or the Vulkan SDK.")
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

    common_opts = ["-DNVDR_TORCH", "-DFW_DO_NOT_OVERRIDE_NEW_DELETE"]

    source_files = [
        "../_cpp/common/common.cpp",
        "../_cpp/common/vkutil.cpp",
        "../_cpp/render/ludus_timestamped_vk.cpp",
        "../_cpp/render/ludus_jpeg.cu",
        "../_cpp/bindings/torch_rasterize_vk.cpp",
    ]
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]

    extra_include_paths = _resolve_vulkan_include()
    ldflags = ["-lcuda", "-lvulkan", "-lnvjpeg"]
    for d in _resolve_vulkan_libdir():
        ldflags.insert(0, f"-L{d}")
        ldflags.insert(0, f"-Wl,-rpath,{d}")

    # Reset CUDA arch list to let PyTorch detect the installed GPU.
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    plugin_name = "ludus_renderer_vk_plugin"

    try:
        lock_fn = os.path.join(
            torch.utils.cpp_extension._get_build_directory(plugin_name, False), "lock"
        )
        if os.path.exists(lock_fn):
            _log.warning("Stale lock file in Vulkan plugin build dir: %s", lock_fn)
    except Exception:
        pass

    _log.info("Compiling Vulkan plugin (this may take a minute on first run)...")
    _cached_plugin = torch.utils.cpp_extension.load(
        name=plugin_name,
        sources=source_paths,
        extra_include_paths=extra_include_paths,
        extra_cflags=common_opts,
        extra_cuda_cflags=common_opts + ["-lineinfo"],
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=True,
    )
    return _cached_plugin
