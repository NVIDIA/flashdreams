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
_EXTENSION_SOURCE = _SOURCE_DIR / "omnidreams_singleview_ext.cpp"
_NATIVE_PRIMITIVES_SOURCE = _SOURCE_DIR / "native_primitives.cpp"
_NATIVE_PRIMITIVES_CUDA_SOURCE = _SOURCE_DIR / "native_primitives_cuda.cu"
_NATIVE_COMMON_HEADER_DIR = _SOURCE_DIR / "native_common"
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
    ]


def _extension_fingerprint_sources() -> list[Path]:
    return [
        *_extension_sources(),
        *sorted(_NATIVE_COMMON_HEADER_DIR.glob("*.h")),
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
            cutlass_include = Path(thirdparty_info["cutlass"]["path"]) / "include"
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
                    extra_include_paths=[str(_SOURCE_DIR), str(cutlass_include)],
                    extra_cflags=[
                        "-O3",
                        "-DOMNIDREAMS_SINGLEVIEW_WITH_CUDA",
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
                        "-DOMNIDREAMS_SINGLEVIEW_WITH_CUDA",
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
