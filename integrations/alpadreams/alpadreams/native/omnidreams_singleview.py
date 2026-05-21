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
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from alpadreams.native.acceleration import (
    NativeAccelerationConfig,
    NativeAvailabilityCheck,
    NativeBackendSelection,
    select_native_extension,
)

_ROOT = Path(__file__).resolve().parents[2] / "omnidreams_singleview"
_NATIVE_BUILD_PATH = _ROOT / "tools" / "native_build.py"
_EXTENSION_SOURCE = _ROOT / "src" / "omnidreams_singleview_ext.cpp"
_NATIVE_MAX_JOBS_ENV = "OMNIDREAMS_SINGLEVIEW_NATIVE_MAX_JOBS"
_PYTORCH_MAX_JOBS_ENV = "MAX_JOBS"
_DEFAULT_NATIVE_MAX_JOBS = "1"

_native_build_module: ModuleType | None = None
_extension: ModuleType | None = None
_extension_load_error: Exception | None = None


def _native_build() -> ModuleType:
    global _native_build_module
    if _native_build_module is not None:
        return _native_build_module

    spec = importlib.util.spec_from_file_location(
        "alpadreams_omnidreams_singleview_native_build",
        _NATIVE_BUILD_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import native build helpers from {_NATIVE_BUILD_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _native_build_module = module
    return module


def validate_thirdparty() -> dict[str, Any]:
    """Validate native source submodules and return their pinned provenance."""

    return {
        name: info.as_dict()
        for name, info in _native_build().validate_thirdparty().items()
    }


def prepare_cutlass_stage(
    build_root: Path | str | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare and return metadata for the patched CUTLASS build stage."""

    return _native_build().prepare_cutlass_stage(
        build_root=build_root,
        force=force,
    ).as_dict()


def build_info(
    build_root: Path | str | None = None,
    *,
    prepare_cutlass: bool = False,
) -> dict[str, Any]:
    """Return native source provenance without compiling the extension."""

    return _native_build().native_provenance(
        build_root=build_root,
        prepare_cutlass=prepare_cutlass,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extension_name(
    stage_info: dict[str, Any],
    thirdparty_info: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(_file_sha256(_EXTENSION_SOURCE).encode("ascii"))
    digest.update(json.dumps(stage_info["manifest"], sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(thirdparty_info, sort_keys=True).encode("utf-8"))
    return f"alpadreams_omnidreams_singleview_native_{digest.hexdigest()[:12]}"


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
    return _validate_max_jobs(
        os.environ.get(_NATIVE_MAX_JOBS_ENV, _DEFAULT_NATIVE_MAX_JOBS)
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


def load_extension(
    build_root: Path | str | None = None,
    *,
    max_jobs: int | str | None = None,
    verbose: bool = False,
) -> ModuleType | None:
    """Compile and load the diagnostic native extension on demand.

    PyTorch's extension builder uses ``MAX_JOBS`` for Ninja fanout. If the caller
    has not already set it, this loader defaults to one compile job to avoid
    surprising memory spikes in local clean builds.

    Returns ``None`` if the extension cannot be built on the current host. The
    full exception is retained and exposed through ``extension_load_error()``.
    """

    global _extension, _extension_load_error
    if _extension is not None:
        return _extension
    if _extension_load_error is not None:
        return None

    try:
        from torch.utils.cpp_extension import load as load_torch_extension

        stage_info = prepare_cutlass_stage(build_root=build_root)
        thirdparty_info = validate_thirdparty()
        extension_name = _extension_name(stage_info, thirdparty_info)
        stage_path = Path(stage_info["path"])
        cutlass_include = stage_path / "include"
        extension_build_dir = stage_path.parent.parent / "torch_extensions" / extension_name
        extension_build_dir.mkdir(parents=True, exist_ok=True)
        manifest = stage_info["manifest"]

        with _scoped_torch_max_jobs(max_jobs):
            _extension = load_torch_extension(
                name=extension_name,
                sources=[str(_EXTENSION_SOURCE)],
                build_directory=str(extension_build_dir),
                extra_include_paths=[str(cutlass_include)],
                extra_cflags=[
                    "-O3",
                    "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SHA="
                    f"\\\"{manifest['cutlass_sha']}\\\"",
                    "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_PATCH_SHA="
                    f"\\\"{manifest['patch_sha256']}\\\"",
                    "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_OVERLAY_SHA="
                    f"\\\"{manifest['overlay_include_sha256']}\\\"",
                    "-DOMNIDREAMS_SINGLEVIEW_SOURCE_SHA="
                    f"\\\"{_file_sha256(_EXTENSION_SOURCE)}\\\"",
                    "-DOMNIDREAMS_SINGLEVIEW_SAGE_ATTENTION_SHA="
                    f"\\\"{thirdparty_info['SageAttention']['head_sha']}\\\"",
                    "-DOMNIDREAMS_SINGLEVIEW_SPARGE_ATTN_SHA="
                    f"\\\"{thirdparty_info['SpargeAttn']['head_sha']}\\\"",
                ],
                with_cuda=False,
                verbose=verbose,
            )
    except Exception as exc:  # pragma: no cover - environment-specific build path
        _extension_load_error = exc
        return None
    return _extension


def extension_load_error() -> Exception | None:
    """Return the last native extension load error, if any."""

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
