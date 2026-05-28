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

"""JIT compilation of the C++/CUDA plugin for Ludus renderer."""

import os
import time

import torch
import torch.utils.cpp_extension
from loguru import logger

_cached_plugin = None


def _get_plugin():
    """Get or compile the C++/CUDA plugin.

    Returns:
        The compiled plugin module.
    """
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and library binaries.
    if os.name == "nt":
        lib_dir = os.path.dirname(__file__) + r"\..\lib"

        def find_cl_path():
            import glob

            def get_sort_key(x):
                x = x.split("\\")[3:]
                x[1] = {
                    "BuildTools": "~0",
                    "Community": "~1",
                    "Pro": "~2",
                    "Professional": "~3",
                    "Enterprise": "~4",
                }.get(x[1], x[1])
                return x

            vs_relative_path = (
                r"\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64"
            )
            paths = glob.glob(r"C:\Program Files" + vs_relative_path)
            paths += glob.glob(r"C:\Program Files (x86)" + vs_relative_path)
            if paths:
                return sorted(paths, key=get_sort_key)[-1]

        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError(
                    "Could not locate a supported Microsoft Visual C++ installation"
                )
            os.environ["PATH"] += ";" + cl_path

    # Compiler options.
    common_defines = ["-DNVDR_TORCH", "-DFW_DO_NOT_OVERRIDE_NEW_DELETE"]
    cc_opts = common_defines + ["-Wall", "-Werror"]
    cuda_opts = common_defines + [
        "-lineinfo",
        "-Xcompiler", "-Wall,-Werror",
        # Suppress nvcc warning about __device__ functions redeclared without
        # __device__ in out-of-line template definitions (Math.hpp MatrixBase).
        "-diag-suppress", "20037",
    ]
    if os.name == "nt":
        cc_opts += ["/wd4067", "/wd4624"]

    # Linker options and source files.
    ldflags: list[str] = []
    if os.name == "posix":
        ldflags = ["-lcuda"]
    source_files = [
        "../_cpp/common/common.cpp",
        "../_cpp/render/ludus_cuda.cu",
        "../_cpp/cudaraster/framework/base/String.cpp",
        "../_cpp/cudaraster/framework/base/Math.cpp",
        "../_cpp/cudaraster/framework/gpu/Buffer.cpp",
        "../_cpp/cudaraster/cudaraster_fw_stub.cpp",
        "../_cpp/cudaraster/CudaRaster.cpp",
        "../_cpp/cudaraster/CudaRasterKernels.cu",
        "../_cpp/bindings/torch_bindings_cuda.cpp",
        "../_cpp/bindings/torch_rasterize_cuda.cpp",
    ]

    # Reset CUDA arch list to let PyTorch detect the installed GPU
    os.environ["TORCH_CUDA_ARCH_LIST"] = ""

    # Check for stale lock files
    plugin_name = "ludus_renderer_plugin"
    build_directory = None
    try:
        build_directory = torch.utils.cpp_extension._get_build_directory(plugin_name, False)
        lock_fn = os.path.join(build_directory, "lock")
        logger.info(
            "Loading Ludus renderer extension {}; build_directory={}.",
            plugin_name,
            build_directory,
        )
        if os.path.exists(lock_fn):
            logger.warning(
                "Ludus renderer extension lock file exists: {}. "
                "If no compiler process is active, this may be a stale PyTorch JIT lock.",
                lock_fn,
            )
    except Exception as exc:
        logger.debug("Could not inspect Ludus renderer extension build directory: {}", exc)

    # Speed up compilation on Windows
    if os.name == "nt":
        os.environ["VSCMD_SKIP_SENDTELEMETRY"] = "1"
        try:
            import distutils._msvccompiler
            import functools

            if not hasattr(distutils._msvccompiler._get_vc_env, "__wrapped__"):
                distutils._msvccompiler._get_vc_env = functools.lru_cache()(
                    distutils._msvccompiler._get_vc_env
                )
        except:
            pass

    # Compile and cache
    compile_started_at = time.perf_counter()
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    extra_include_paths = [
        os.path.join(os.path.dirname(__file__), "../_cpp/cudaraster/framework"),
    ]
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
    logger.info(
        "Loaded Ludus renderer extension {} in {:.1f}s{}.",
        plugin_name,
        time.perf_counter() - compile_started_at,
        f"; build_directory={build_directory}" if build_directory else "",
    )

    return _cached_plugin


def get_log_level():
    """Get current log level.

    Returns:
        Current log level. See ``set_log_level()`` for possible values.
    """
    return _get_plugin().get_log_level()


def set_log_level(level):
    """Set log level.

    Log levels follow the convention on the C++ side of Torch:
        0 = Info,
        1 = Warning,
        2 = Error,
        3 = Fatal.
    The default log level is 1.

    Args:
        level: New log level as integer.
    """
    _get_plugin().set_log_level(level)
