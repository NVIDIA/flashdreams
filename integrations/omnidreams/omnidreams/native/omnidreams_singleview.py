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

"""Lazy native extension loading for OmniDreams single-view acceleration."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from omnidreams.native.acceleration import (
    NativeAccelerationConfig,
    NativeAvailabilityCheck,
    NativeBackendSelection,
    select_native_extension,
)

_ROOT = Path(__file__).resolve().parents[2] / "omnidreams_singleview"
_NATIVE_BUILD_PATH = _ROOT / "tools" / "native_build.py"
_SOURCE_DIR = _ROOT / "src"
_PYTHON_DIR = _ROOT / "python"
_EXTENSION_SOURCE = _SOURCE_DIR / "omnidreams_singleview_ext.cpp"
_NATIVE_PRIMITIVES_SOURCE = _SOURCE_DIR / "native_primitives.cpp"
_NATIVE_PRIMITIVES_CUDA_SOURCE = _SOURCE_DIR / "native_primitives_cuda.cu"
_NATIVE_COMMON_HEADER_DIR = _SOURCE_DIR / "native_common"
_DIT_STREAMING_DIR = _SOURCE_DIR / "dit_streaming"
_DIT_STREAMING_KERNEL_DIR = _DIT_STREAMING_DIR / "kernels"
_DIT_STREAMING_PYEXT_DIR = _DIT_STREAMING_DIR / "pyext"
_DIT_STREAMING_COMMON_DIR = _DIT_STREAMING_DIR / "common"
_VAE_STREAMING_DIR = _SOURCE_DIR / "vae_streaming"
_PYTORCH_MAX_JOBS_ENV = "MAX_JOBS"
_DEFAULT_MAX_JOBS_CAP = 8
_NATIVE_CUDA_ARCH_LIST_ENV = "OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST"
_PYTORCH_CUDA_ARCH_LIST_ENV = "TORCH_CUDA_ARCH_LIST"
_DEFAULT_CUDA_ARCH_LIST = "12.0a"

_native_build_module: ModuleType | None = None
_extension: ModuleType | None = None
_extension_load_error: Exception | None = None
_state_lock = threading.RLock()


def _native_build() -> ModuleType:
    global _native_build_module
    with _state_lock:
        if _native_build_module is not None:
            return _native_build_module

        spec = importlib.util.spec_from_file_location(
            "omnidreams_singleview_native_build",
            _NATIVE_BUILD_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot import native build helpers from {_NATIVE_BUILD_PATH}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _native_build_module = module
        return module


def validate_thirdparty() -> dict[str, Any]:
    """Validate native source checkouts and return their pinned provenance."""

    return {
        name: info.as_dict()
        for name, info in _native_build().validate_thirdparty().items()
    }


def sync_thirdparty(*, force: bool = False) -> dict[str, Any]:
    """Synchronize native source checkouts and return their pinned provenance."""

    return {
        name: info.as_dict()
        for name, info in _native_build().sync_thirdparty(force=force).items()
    }


def load_python_module(name: str) -> ModuleType:
    """Load a helper module shipped with the single-view native sources."""

    if not name.isidentifier():
        raise ValueError(
            f"Native helper module name must be an identifier, got {name!r}"
        )
    path = _PYTHON_DIR / f"{name}.py"
    if not path.is_file():
        raise ImportError(f"Unknown OmniDreams single-view native helper {name!r}")
    module_name = f"omnidreams_singleview_native_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import OmniDreams single-view native helper from {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    python_dir = str(_PYTHON_DIR)
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    spec.loader.exec_module(module)
    return module


def _first_library_dir(library_name: str, candidates: tuple[Path, ...]) -> Path | None:
    for directory in candidates:
        if (directory / library_name).is_file():
            return directory
    return None


def build_info(
    build_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return native source provenance without compiling the extension."""

    return _native_build().native_provenance(build_root=build_root)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extension_sources() -> list[Path]:
    return [
        _EXTENSION_SOURCE,
        _NATIVE_PRIMITIVES_SOURCE,
        _NATIVE_PRIMITIVES_CUDA_SOURCE,
        _DIT_STREAMING_DIR / "streaming_dit_bindings.cpp",
        _VAE_STREAMING_DIR / "vae_streaming_bindings.cpp",
        _VAE_STREAMING_DIR / "lightvae_ops.cu",
        _VAE_STREAMING_DIR / "lightvae_fp8_ops.cu",
        _VAE_STREAMING_DIR / "lightvae_fp8_direct_stages.cu",
        _VAE_STREAMING_DIR / "lightvae_fp8_warp_mma_stages.cu",
        _VAE_STREAMING_DIR / "lightvae_fp8_attention.cu",
        _DIT_STREAMING_PYEXT_DIR / "streaming_dit_bridge.cu",
        _DIT_STREAMING_PYEXT_DIR / "sage3_blackwell_api_shim.cu",
        _DIT_STREAMING_PYEXT_DIR / "sage3_fp4_quant_shim.cu",
        _DIT_STREAMING_KERNEL_DIR / "attention.cu",
        _DIT_STREAMING_KERNEL_DIR / "block_quant.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_adaln_lora.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_block.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_fp8_flash.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_fp8_flash_tc.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_fp8_tc_probe.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_fp8_two_gemm.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_gemm_bf16.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_modulate.cu",
        _DIT_STREAMING_KERNEL_DIR / "ops.cu",
        _DIT_STREAMING_KERNEL_DIR / "sage3_attention.cu",
        _DIT_STREAMING_KERNEL_DIR / "sparge_attention_sm89_inst.cu",
        _DIT_STREAMING_KERNEL_DIR / "transformer_block.cu",
    ]


def _extension_fingerprint_sources() -> list[Path]:
    return [
        *_extension_sources(),
        *sorted(_NATIVE_COMMON_HEADER_DIR.glob("*.h")),
        *sorted(_DIT_STREAMING_DIR.rglob("*.h")),
        *sorted(_DIT_STREAMING_DIR.rglob("*.cuh")),
        *sorted(_DIT_STREAMING_DIR.rglob("*.hpp")),
        *sorted(_VAE_STREAMING_DIR.rglob("*.h")),
        *sorted(_VAE_STREAMING_DIR.rglob("*.hpp")),
        *sorted(_PYTHON_DIR.glob("*.py")),
    ]


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for source in _extension_fingerprint_sources():
        digest.update(source.relative_to(_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(source).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _extension_name(thirdparty_info: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_source_fingerprint().encode("ascii"))
    digest.update(json.dumps(thirdparty_info, sort_keys=True).encode("utf-8"))
    return f"omnidreams_singleview_native_{digest.hexdigest()[:12]}"


def _validate_max_jobs(value: int | str) -> str:
    text = str(value).strip()
    try:
        jobs = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Native max jobs must be a positive integer, got {value!r}"
        ) from exc
    if jobs < 1:
        raise ValueError(f"Native max jobs must be a positive integer, got {value!r}")
    return str(jobs)


def _resolved_max_jobs(max_jobs: int | str | None) -> str | None:
    if max_jobs is not None:
        return _validate_max_jobs(max_jobs)
    if os.environ.get(_PYTORCH_MAX_JOBS_ENV):
        return None
    return str(min(os.cpu_count() or 1, _DEFAULT_MAX_JOBS_CAP))


def _resolved_cuda_arch_list() -> str | None:
    if os.environ.get(_PYTORCH_CUDA_ARCH_LIST_ENV):
        return None
    return os.environ.get(_NATIVE_CUDA_ARCH_LIST_ENV, _DEFAULT_CUDA_ARCH_LIST)


def _effective_cuda_arch_list() -> str:
    return os.environ.get(
        _PYTORCH_CUDA_ARCH_LIST_ENV,
        os.environ.get(_NATIVE_CUDA_ARCH_LIST_ENV, _DEFAULT_CUDA_ARCH_LIST),
    )


def _python_package_dir(package: str) -> Path | None:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        return None
    locations = list(spec.submodule_search_locations)
    if not locations:
        return None
    return Path(locations[0])


@contextlib.contextmanager
def _scoped_torch_max_jobs(max_jobs: int | str | None) -> Iterator[None]:
    resolved = _resolved_max_jobs(max_jobs)
    if resolved is None:
        yield
        return

    previous = os.environ.get(_PYTORCH_MAX_JOBS_ENV)
    os.environ[_PYTORCH_MAX_JOBS_ENV] = resolved
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_PYTORCH_MAX_JOBS_ENV, None)
        else:
            os.environ[_PYTORCH_MAX_JOBS_ENV] = previous


@contextlib.contextmanager
def _scoped_cuda_arch_list() -> Iterator[None]:
    resolved = _resolved_cuda_arch_list()
    if resolved is None:
        yield
        return

    previous = os.environ.get(_PYTORCH_CUDA_ARCH_LIST_ENV)
    os.environ[_PYTORCH_CUDA_ARCH_LIST_ENV] = resolved
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_PYTORCH_CUDA_ARCH_LIST_ENV, None)
        else:
            os.environ[_PYTORCH_CUDA_ARCH_LIST_ENV] = previous


def load_extension(
    build_root: Path | str | None = None,
    *,
    max_jobs: int | str | None = None,
    verbose: bool = False,
) -> ModuleType | None:
    """Compile and load the CUDA native extension on demand.

    PyTorch's extension builder uses ``MAX_JOBS`` for Ninja fanout. If the caller
    has not already set it, this loader sets a modest default cap to avoid
    runaway memory use in local clean builds.

    Returns ``None`` if the extension cannot be built on the current host. The
    full exception is retained and exposed through ``extension_load_error()``.
    """

    global _extension, _extension_load_error
    with _state_lock:
        if _extension is not None:
            return _extension
        _extension_load_error = None

        try:
            from torch.utils.cpp_extension import load as load_torch_extension

            thirdparty_info = validate_thirdparty()
            extension_name = _extension_name(thirdparty_info)
            cutlass_dir = Path(thirdparty_info["cutlass"]["path"])
            cutlass_include = cutlass_dir / "include"
            sage_attention_dir = Path(thirdparty_info["SageAttention"]["path"])
            sparge_attn_dir = Path(thirdparty_info["SpargeAttn"]["path"])
            sparge_attn_csrc = sparge_attn_dir / "csrc"
            cudnn_frontend_include = (
                Path(thirdparty_info["cudnn-frontend"]["path"]) / "include"
            )
            cudnn_package_dir = _python_package_dir("nvidia.cudnn")
            cudnn_include = (
                cudnn_package_dir / "include"
                if cudnn_package_dir is not None
                and (cudnn_package_dir / "include" / "cudnn.h").is_file()
                else None
            )
            cudnn_lib = (
                cudnn_package_dir / "lib"
                if cudnn_package_dir is not None
                and (cudnn_package_dir / "lib" / "libcudnn.so.9").is_file()
                else None
            )
            cuda_driver_lib = _first_library_dir(
                "libcuda.so",
                (
                    Path("/usr/lib/wsl/lib"),
                    Path("/usr/lib/x86_64-linux-gnu"),
                    Path("/usr/local/cuda/lib64"),
                ),
            )
            extension_build_dir = _native_build().torch_extension_build_dir(
                extension_name,
                build_root=build_root,
            )
            extension_build_dir.mkdir(parents=True, exist_ok=True)

            with _scoped_torch_max_jobs(max_jobs), _scoped_cuda_arch_list():
                _extension = load_torch_extension(
                    name=extension_name,
                    sources=[str(source) for source in _extension_sources()],
                    build_directory=str(extension_build_dir),
                    extra_include_paths=[
                        str(_SOURCE_DIR),
                        str(_DIT_STREAMING_DIR),
                        str(_DIT_STREAMING_KERNEL_DIR),
                        str(_DIT_STREAMING_PYEXT_DIR),
                        str(_DIT_STREAMING_COMMON_DIR),
                        str(_VAE_STREAMING_DIR),
                        str(cutlass_include),
                        str(cutlass_dir / "tools" / "util" / "include"),
                        str(cutlass_dir / "examples" / "common"),
                        str(cutlass_dir / "examples" / "41_fused_multi_head_attention"),
                        str(sage_attention_dir),
                        str(
                            sage_attention_dir
                            / "sageattention3_blackwell"
                            / "sageattn3"
                        ),
                        str(
                            sage_attention_dir
                            / "sageattention3_blackwell"
                            / "sageattn3"
                            / "blackwell"
                        ),
                        str(
                            sage_attention_dir
                            / "sageattention3_blackwell"
                            / "sageattn3"
                            / "quantization"
                        ),
                        str(sparge_attn_csrc),
                        str(sparge_attn_csrc / "qattn"),
                        str(sparge_attn_csrc / "fused"),
                        str(cudnn_frontend_include),
                        *([] if cudnn_include is None else [str(cudnn_include)]),
                    ],
                    extra_cflags=[
                        "-O3",
                        "-std=c++20",
                        "-DOMNIDREAMS_SINGLEVIEW_WITH_CUDA",
                        "-DOMNIDREAMS_SINGLEVIEW_USE_CUTLASS",
                        "-DOMNIDREAMS_SINGLEVIEW_HAS_SAGE3=1",
                        "-DOMNIDREAMS_SINGLEVIEW_HAS_SPARGE=1",
                        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SHA="
                        f'\\"{thirdparty_info["cutlass"]["commit"]}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SOURCE_SHA="
                        f'\\"{thirdparty_info["cutlass"]["source_sha256"]}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_SOURCE_SHA="
                        f'\\"{_file_sha256(_EXTENSION_SOURCE)}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_SOURCE_FINGERPRINT_SHA="
                        f'\\"{_source_fingerprint()}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_NATIVE_PRIMITIVES_SOURCE_SHA="
                        f'\\"{_file_sha256(_NATIVE_PRIMITIVES_SOURCE)}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_CUDA_SOURCE_SHA="
                        f'\\"{_file_sha256(_NATIVE_PRIMITIVES_CUDA_SOURCE)}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_SAGE_ATTENTION_SHA="
                        f'\\"{thirdparty_info["SageAttention"]["commit"]}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_SPARGE_ATTN_SHA="
                        f'\\"{thirdparty_info["SpargeAttn"]["commit"]}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST="
                        f'\\"{_effective_cuda_arch_list()}\\"',
                    ],
                    extra_cuda_cflags=[
                        "-O3",
                        "-std=c++20",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "-lineinfo",
                        "--use_fast_math",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        "-DQBLKSIZE=128",
                        "-DKBLKSIZE=128",
                        "-DCTA256",
                        "-DDQINRMEM",
                        "-DEXECMODE=0",
                        "-DNDEBUG",
                        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                        "-DOMNIDREAMS_SINGLEVIEW_WITH_CUDA",
                        "-DOMNIDREAMS_SINGLEVIEW_USE_CUTLASS",
                        "-DOMNIDREAMS_SINGLEVIEW_HAS_SAGE3=1",
                        "-DOMNIDREAMS_SINGLEVIEW_HAS_SPARGE=1",
                    ],
                    extra_ldflags=[
                        "-lcublas",
                        "-lcublasLt",
                        *([] if cudnn_lib is None else [f"-L{cudnn_lib}"]),
                        "-lcudnn" if cudnn_lib is None else "-l:libcudnn.so.9",
                        *([] if cuda_driver_lib is None else [f"-L{cuda_driver_lib}"]),
                        "-lcuda",
                        "-lnvrtc",
                    ],
                    with_cuda=True,
                    verbose=verbose,
                )
        except Exception as exc:  # pragma: no cover - environment-specific build path
            _extension_load_error = exc
            return None
        return _extension


def extension_load_error() -> Exception | None:
    """Return the last native extension load error, if any."""

    with _state_lock:
        return _extension_load_error


def select_backend(
    component: str,
    config: NativeAccelerationConfig | None = None,
    *,
    availability_check: NativeAvailabilityCheck | None = None,
) -> NativeBackendSelection:
    """Resolve OmniDreams single-view native use for a pipeline component."""

    return select_native_extension(
        config or NativeAccelerationConfig(),
        component=component,
        extension_loader=load_extension,
        extension_error=extension_load_error,
        availability_check=availability_check,
    )
