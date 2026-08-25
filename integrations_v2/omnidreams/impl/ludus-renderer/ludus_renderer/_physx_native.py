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

"""Build and load Ludus's standalone PhysX extension."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import torch

_PHYSX_VERSION = "5.9.0"
_PHYSX_COMMIT = "517a0073715120e114ee055b63b26c95e00d9039"
_PHYSX_ARCHIVE_URL = (
    f"https://github.com/NVIDIA-Omniverse/PhysX/archive/{_PHYSX_COMMIT}.zip"
)
_PHYSX_ARCHIVE_SHA256 = (
    "b2f713bac94e2655614b1c8c34ba18750673d5d780fba19e0a23a21bda5695bb"
)
_MODULE_NAME = "ludus_physx_native"
_BUILD_LOCK_TIMEOUT_SECONDS = 1_800.0
_BUILD_LOCK_STALE_SECONDS = 1_800.0
_BUILD_LOCK_HEARTBEAT_SECONDS = 60.0
_CACHED_MODULE: ModuleType | None = None


def _cache_root() -> Path:
    override = os.environ.get("LUDUS_PHYSX_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "ludus-renderer" / f"physx-{_PHYSX_VERSION}"


def _download_source(cache_root: Path) -> Path:
    source_root = cache_root / "source" / f"PhysX-{_PHYSX_COMMIT}" / "physx"
    if source_root.is_dir():
        return source_root
    cache_root.mkdir(parents=True, exist_ok=True)
    archive = cache_root / f"PhysX-{_PHYSX_VERSION}.zip"
    if not archive.is_file() or _sha256(archive) != _PHYSX_ARCHIVE_SHA256:
        partial = archive.with_suffix(".zip.partial")
        partial.unlink(missing_ok=True)
        urllib.request.urlretrieve(_PHYSX_ARCHIVE_URL, partial)
        digest = _sha256(partial)
        if digest != _PHYSX_ARCHIVE_SHA256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                "PhysX source archive failed SHA-256 verification: "
                f"expected {_PHYSX_ARCHIVE_SHA256}, received {digest}"
            )
        partial.replace(archive)
    extraction = cache_root / "source.partial"
    if extraction.exists():
        shutil.rmtree(extraction)
    extraction.mkdir()
    with zipfile.ZipFile(archive) as source_zip:
        source_zip.extractall(extraction)
    destination = cache_root / "source"
    if destination.exists():
        shutil.rmtree(destination)
    extraction.replace(destination)
    if not source_root.is_dir():
        raise RuntimeError("PhysX archive did not contain the expected source tree")
    return source_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_source_dir() -> Path:
    return Path(__file__).resolve().parent / "_cpp" / "physx"


def _heartbeat_build_lock(
    lock_path: Path, owner: str, stopped: threading.Event
) -> None:
    """Keep a live build lock from being mistaken for an abandoned one."""
    while not stopped.wait(_BUILD_LOCK_HEARTBEAT_SECONDS):
        try:
            if lock_path.read_text(encoding="utf-8") != owner:
                return
            lock_path.touch()
        except OSError:
            return


@contextmanager
def _build_lock(cache_root: Path) -> Iterator[None]:
    """Serialize first-use builds that share the platform cache."""
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / "build.lock"
    deadline: float | None = None
    descriptor: int | None = None
    owner = f"{os.getpid()}:{uuid.uuid4().hex}\n"
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.write(descriptor, owner.encode())
        except FileExistsError:
            if deadline is None:
                # Start the wait timeout only after observing the competing lock.
                # A lock created between entering this function and the first open
                # attempt must get a full stale interval before we time out.
                deadline = time.monotonic() + _BUILD_LOCK_TIMEOUT_SECONDS
            try:
                stale = (
                    time.time() - lock_path.stat().st_mtime >= _BUILD_LOCK_STALE_SECONDS
                )
            except FileNotFoundError:
                continue
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for the PhysX build lock at {lock_path}"
                ) from None
            time.sleep(0.1)
    heartbeat_stopped = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_build_lock,
        args=(lock_path, owner, heartbeat_stopped),
        name="ludus-physx-build-lock-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        yield
    finally:
        heartbeat_stopped.set()
        heartbeat.join()
        os.close(descriptor)
        try:
            if lock_path.read_text(encoding="utf-8") == owner:
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _module_path(output_dir: Path) -> Path | None:
    suffixes = ("*.pyd",) if os.name == "nt" else ("*.so",)
    candidates = [path for pattern in suffixes for path in output_dir.glob(pattern)]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _discard_relocated_cmake_build(build_dir: Path, source_dir: Path) -> None:
    """Remove generated CMake state that belongs to a different checkout."""
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        return
    prefix = "CMAKE_HOME_DIRECTORY:INTERNAL="
    try:
        cached_source = next(
            line.removeprefix(prefix)
            for line in cache_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.startswith(prefix)
        )
    except StopIteration:
        return
    cached_normalized = os.path.normcase(os.path.abspath(cached_source))
    current_normalized = os.path.normcase(os.path.abspath(source_dir))
    if cached_normalized != current_normalized:
        shutil.rmtree(build_dir)


def _configure_and_build(cache_root: Path, physx_root: Path) -> Path:
    """Configure once for this process and let CMake decide what to rebuild."""
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError(
            "ludus-renderer requires the CMake Python package to build native PhysX"
        )
    output_dir = cache_root / "module"
    build_dir = cache_root / f"build-{platform.system().lower()}-{platform.machine()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = _native_source_dir()
    _discard_relocated_cmake_build(build_dir, source_dir)
    torch_include = Path(torch.__file__).resolve().parent / "include"
    configure = [
        cmake,
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DPHYSX_ROOT_DIR={physx_root.as_posix()}",
        f"-DPYBIND11_INCLUDE_DIR={torch_include.as_posix()}",
        f"-DLUDUS_MODULE_OUTPUT_DIR={output_dir.as_posix()}",
    ]
    if os.name == "nt":
        dummy_freeglut = cache_root / "dummy-freeglut" / "win64"
        dummy_freeglut.mkdir(parents=True, exist_ok=True)
        for name in ("freeglut.dll", "freeglutd.dll"):
            (dummy_freeglut / name).touch()
        configure.extend(
            [
                "-G",
                "Visual Studio 17 2022",
                "-A",
                "x64",
                f"-DPHYSX_SLN_FREEGLUT_PATH={dummy_freeglut.parent.as_posix()}",
                "-DCMAKE_CONFIGURATION_TYPES=Release",
            ]
        )
    else:
        configure.extend(["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"])
    build_environment = os.environ.copy()
    build_environment["MSBUILDDISABLENODEREUSE"] = "1"
    subprocess.run(configure, check=True, env=build_environment)
    build = [
        cmake,
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "--target",
        _MODULE_NAME,
        "--parallel",
        str(min(16, os.cpu_count() or 1)),
    ]
    subprocess.run(build, check=True, env=build_environment)
    module_path = _module_path(output_dir)
    if module_path is None:
        raise RuntimeError("native PhysX build completed without producing a module")
    return module_path


def load_native_physx() -> ModuleType:
    """Build once and return the standalone Ludus PhysX module."""
    global _CACHED_MODULE
    if _CACHED_MODULE is not None:
        return _CACHED_MODULE
    cache_root = _cache_root()
    with _build_lock(cache_root):
        # Another thread may have loaded the module while this thread waited.
        if _CACHED_MODULE is not None:
            return _CACHED_MODULE
        physx_root = _download_source(cache_root)
        # Always enter CMake on the first load in a process. Its generated
        # dependency graph, rather than a partial Python timestamp check,
        # determines whether the cached native target is up to date.
        module_path = _configure_and_build(cache_root, physx_root)
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load native PhysX module at {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        _CACHED_MODULE = module
    return module


__all__ = ["load_native_physx"]
