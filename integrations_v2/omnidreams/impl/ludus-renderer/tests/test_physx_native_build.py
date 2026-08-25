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

"""CPU-only contracts for the first-use PhysX CMake build."""

import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from ludus_renderer import _physx_native

pytestmark = pytest.mark.ci_cpu


def test_build_lock_can_become_stale_before_wait_timeout() -> None:
    assert (
        _physx_native._BUILD_LOCK_STALE_SECONDS
        <= _physx_native._BUILD_LOCK_TIMEOUT_SECONDS
    )
    assert (
        _physx_native._BUILD_LOCK_HEARTBEAT_SECONDS
        < _physx_native._BUILD_LOCK_STALE_SECONDS
    )


def test_build_lock_reclaims_abandoned_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "build.lock"
    lock_path.write_text("abandoned\n", encoding="utf-8")
    stale_time = time.time() - _physx_native._BUILD_LOCK_STALE_SECONDS - 1.0
    os.utime(lock_path, (stale_time, stale_time))

    with _physx_native._build_lock(tmp_path):
        assert lock_path.read_text(encoding="utf-8").startswith(f"{os.getpid()}:")

    assert not lock_path.exists()


def test_build_lock_recovers_lock_created_after_waiter_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock_path = tmp_path / "build.lock"
    clock = 0.0
    original_open = os.open
    racing_process_created_lock = False

    def racing_open(path: Path, flags: int, mode: int) -> int:
        nonlocal clock, racing_process_created_lock
        if not racing_process_created_lock:
            racing_process_created_lock = True
            clock = 1.0
            descriptor = original_open(path, flags, mode)
            os.write(descriptor, b"abandoned\n")
            os.close(descriptor)
            os.utime(path, (1_001.0, 1_001.0))
            raise FileExistsError
        return original_open(path, flags, mode)

    def advance_clock(_seconds: float) -> None:
        nonlocal clock
        clock += 1.0

    monkeypatch.setattr(_physx_native, "_BUILD_LOCK_TIMEOUT_SECONDS", 10.0)
    monkeypatch.setattr(_physx_native, "_BUILD_LOCK_STALE_SECONDS", 10.0)
    monkeypatch.setattr(_physx_native.os, "open", racing_open)
    monkeypatch.setattr(_physx_native.time, "monotonic", lambda: clock)
    monkeypatch.setattr(_physx_native.time, "time", lambda: 1_000.0 + clock)
    monkeypatch.setattr(_physx_native.time, "sleep", advance_clock)

    with _physx_native._build_lock(tmp_path):
        assert lock_path.read_text(encoding="utf-8").startswith(f"{os.getpid()}:")

    assert not lock_path.exists()


def test_build_lock_heartbeat_refreshes_active_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_physx_native, "_BUILD_LOCK_HEARTBEAT_SECONDS", 0.01)

    with _physx_native._build_lock(tmp_path):
        lock_path = tmp_path / "build.lock"
        old_time = time.time() - _physx_native._BUILD_LOCK_STALE_SECONDS - 1.0
        os.utime(lock_path, (old_time, old_time))
        stale_mtime = lock_path.stat().st_mtime
        deadline = time.monotonic() + 1.0
        while lock_path.stat().st_mtime == stale_mtime and time.monotonic() < deadline:
            time.sleep(0.01)

        assert lock_path.stat().st_mtime > stale_mtime

    assert not lock_path.exists()


def test_load_native_physx_runs_cmake_path_once_per_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module_path = tmp_path / "ludus_physx_native.pyd"
    module_path.touch()
    loaded_module = ModuleType(_physx_native._MODULE_NAME)
    configure_calls: list[tuple[Path, Path]] = []

    @contextmanager
    def fake_build_lock(cache_root: Path) -> Any:
        yield

    def fake_configure(cache_root: Path, physx_root: Path) -> Path:
        configure_calls.append((cache_root, physx_root))
        return module_path

    loader = SimpleNamespace(exec_module=lambda module: None)
    spec = SimpleNamespace(loader=loader)
    monkeypatch.setattr(_physx_native, "_CACHED_MODULE", None)
    monkeypatch.setattr(_physx_native, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(_physx_native, "_build_lock", fake_build_lock)
    monkeypatch.setattr(
        _physx_native, "_download_source", lambda cache_root: tmp_path / "physx"
    )
    monkeypatch.setattr(_physx_native, "_configure_and_build", fake_configure)
    monkeypatch.setattr(
        _physx_native.importlib.util,
        "spec_from_file_location",
        lambda name, path: spec,
    )
    monkeypatch.setattr(
        _physx_native.importlib.util,
        "module_from_spec",
        lambda loaded_spec: loaded_module,
    )
    monkeypatch.delitem(
        _physx_native.sys.modules, _physx_native._MODULE_NAME, raising=False
    )

    first = _physx_native.load_native_physx()
    second = _physx_native.load_native_physx()

    assert first is loaded_module
    assert second is loaded_module
    assert configure_calls == [(tmp_path, tmp_path / "physx")]


def test_configure_and_build_delegates_freshness_to_cmake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing_module = tmp_path / "module" / "ludus_physx_native.pyd"
    existing_module.parent.mkdir()
    existing_module.touch()
    commands: list[list[str]] = []

    monkeypatch.setattr(_physx_native.shutil, "which", lambda command: "cmake")
    monkeypatch.setattr(
        _physx_native,
        "_module_path",
        lambda output_dir: existing_module,
    )
    monkeypatch.setattr(
        _physx_native.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    result = _physx_native._configure_and_build(tmp_path, tmp_path / "physx")

    assert result == existing_module
    assert len(commands) == 2
    assert commands[0][:2] == ["cmake", "-S"]
    assert "-B" in commands[0]
    assert commands[1][:2] == ["cmake", "--build"]
    assert "--target" in commands[1]
    assert _physx_native._MODULE_NAME in commands[1]


def test_discard_relocated_cmake_build_removes_stale_checkout(
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        "CMAKE_HOME_DIRECTORY:INTERNAL=C:/old/flashdreams/ludus_renderer/_cpp/physx\n",
        encoding="utf-8",
    )

    _physx_native._discard_relocated_cmake_build(
        build_dir, tmp_path / "current" / "_cpp" / "physx"
    )

    assert not build_dir.exists()


def test_discard_relocated_cmake_build_preserves_current_checkout(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "current" / "_cpp" / "physx"
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source_dir.as_posix()}\n",
        encoding="utf-8",
    )

    _physx_native._discard_relocated_cmake_build(build_dir, source_dir)

    assert build_dir.exists()
