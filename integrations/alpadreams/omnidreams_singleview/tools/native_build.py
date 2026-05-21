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

"""Build preparation helpers for the OmniDreams single-view native path."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]

THIRDPARTY_DIR = ROOT / "3rdparty"
CUTLASS_DIR = THIRDPARTY_DIR / "cutlass"
SAGE_ATTENTION_DIR = THIRDPARTY_DIR / "SageAttention"
SPARGE_ATTN_DIR = THIRDPARTY_DIR / "SpargeAttn"

PATCH_DIR = ROOT / "patches" / "cutlass"
CUTLASS_PATCH = PATCH_DIR / "sm120-tma-pool.patch"
CUTLASS_OVERLAY_INCLUDE = PATCH_DIR / "include"

_DEFAULT_BUILD_ROOT_ENV = "OMNIDREAMS_SINGLEVIEW_NATIVE_BUILD_ROOT"
_STAGE_SCHEMA_VERSION = 1


class NativeBuildError(RuntimeError):
    """Raised when native build preparation cannot safely continue."""


@dataclass(frozen=True)
class SubmoduleInfo:
    """Validated source submodule state."""

    name: str
    path: Path
    relative_path: str
    recorded_sha: str
    head_sha: str
    tracked_status: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.relative_path,
            "recorded_sha": self.recorded_sha,
            "head_sha": self.head_sha,
            "tracked_status": list(self.tracked_status),
        }


@dataclass(frozen=True)
class CutlassStageInfo:
    """Prepared CUTLASS stage metadata."""

    path: Path
    manifest_path: Path
    manifest: Mapping[str, object]
    reused: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "manifest_path": str(self.manifest_path),
            "manifest": dict(self.manifest),
            "reused": self.reused,
        }


def _run_git(cwd: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip()
        raise NativeBuildError(f"git {' '.join(args)} failed in {cwd}: {message}")
    return proc.stdout.strip()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _submodule_head(path: Path) -> str:
    try:
        return _run_git(path, ["rev-parse", "HEAD"])
    except NativeBuildError as exc:
        raise NativeBuildError(
            f"{_repo_relative(path)} is not an initialized git submodule"
        ) from exc


def _recorded_gitlink(relative_path: str) -> str:
    try:
        return _run_git(REPO_ROOT, ["rev-parse", f"HEAD:{relative_path}"])
    except NativeBuildError as exc:
        raise NativeBuildError(
            f"{relative_path} is not recorded as a submodule in the parent commit"
        ) from exc


def _tracked_status(path: Path) -> tuple[str, ...]:
    output = _run_git(path, ["status", "--short", "--untracked-files=no"])
    return tuple(line for line in output.splitlines() if line)


def validate_submodule(name: str, path: Path) -> SubmoduleInfo:
    """Validate that a source submodule is initialized, clean, and pinned."""

    if not path.exists():
        raise NativeBuildError(f"Missing required source submodule: {path}")

    relative_path = _repo_relative(path)
    head_sha = _submodule_head(path)
    recorded_sha = _recorded_gitlink(relative_path)
    tracked_status = _tracked_status(path)

    if head_sha != recorded_sha:
        raise NativeBuildError(
            f"{relative_path} is checked out at {head_sha}, "
            f"but the parent commit records {recorded_sha}"
        )
    if tracked_status:
        details = "\n".join(tracked_status)
        raise NativeBuildError(
            f"{relative_path} has tracked-file modifications:\n{details}"
        )

    return SubmoduleInfo(
        name=name,
        path=path,
        relative_path=relative_path,
        recorded_sha=recorded_sha,
        head_sha=head_sha,
        tracked_status=tracked_status,
    )


def validate_thirdparty() -> dict[str, SubmoduleInfo]:
    """Validate all source submodules required by the native path."""

    return {
        "cutlass": validate_submodule("cutlass", CUTLASS_DIR),
        "SageAttention": validate_submodule("SageAttention", SAGE_ATTENTION_DIR),
        "SpargeAttn": validate_submodule("SpargeAttn", SPARGE_ATTN_DIR),
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()

    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_hash_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def cutlass_patch_fingerprint() -> dict[str, str]:
    """Return the content hashes that determine the patched CUTLASS stage."""

    return {
        "patch_sha256": _hash_file(CUTLASS_PATCH),
        "overlay_include_sha256": _hash_tree(CUTLASS_OVERLAY_INCLUDE),
    }


def _default_build_root() -> Path:
    override = os.environ.get(_DEFAULT_BUILD_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return ROOT / "build"


def _expected_manifest(cutlass_info: SubmoduleInfo) -> dict[str, object]:
    return {
        "schema_version": _STAGE_SCHEMA_VERSION,
        "cutlass_sha": cutlass_info.head_sha,
        **cutlass_patch_fingerprint(),
    }


def _read_manifest(path: Path) -> dict[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise NativeBuildError(f"Invalid CUTLASS stage manifest at {path}: {exc}")

    if not isinstance(data, dict):
        raise NativeBuildError(f"Invalid CUTLASS stage manifest at {path}: not a dict")
    return data


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


@contextlib.contextmanager
def _stage_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _extract_git_archive(source: Path, destination: Path) -> None:
    with subprocess.Popen(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        assert proc.stdout is not None
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
        assert proc.stderr is not None
        stderr = proc.stderr.read()
        proc.wait()
        if proc.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise NativeBuildError(f"git archive failed for {source}: {message}")


def _apply_patch(stage_path: Path, patch_path: Path) -> None:
    for args in (
        ["apply", "--check", str(patch_path)],
        ["apply", str(patch_path)],
    ):
        try:
            _run_git(stage_path, args)
        except NativeBuildError as exc:
            raise NativeBuildError(
                f"Failed to apply CUTLASS patch {patch_path} in {stage_path}"
            ) from exc


def _refresh_cutlass_stage(stage_path: Path) -> None:
    if stage_path.exists():
        shutil.rmtree(stage_path)
    stage_path.mkdir(parents=True)
    _extract_git_archive(CUTLASS_DIR, stage_path)
    _apply_patch(stage_path, CUTLASS_PATCH)
    shutil.copytree(
        CUTLASS_OVERLAY_INCLUDE,
        stage_path / "include",
        dirs_exist_ok=True,
    )


def prepare_cutlass_stage(
    build_root: Path | str | None = None,
    *,
    force: bool = False,
) -> CutlassStageInfo:
    """Prepare a patched CUTLASS copy without modifying the submodule worktree."""

    thirdparty = validate_thirdparty()
    cutlass_info = thirdparty["cutlass"]
    manifest = _expected_manifest(cutlass_info)

    resolved_build_root = Path(build_root).resolve() if build_root else _default_build_root()
    deps_root = resolved_build_root / "_deps"
    stage_path = deps_root / "cutlass-patched"
    manifest_path = stage_path / ".omnidreams_cutlass_stage.json"
    lock_path = deps_root / ".cutlass-stage.lock"

    with _stage_lock(lock_path):
        existing_manifest = _read_manifest(manifest_path)
        if not force and stage_path.exists() and existing_manifest == manifest:
            return CutlassStageInfo(
                path=stage_path,
                manifest_path=manifest_path,
                manifest=manifest,
                reused=True,
            )

        _refresh_cutlass_stage(stage_path)
        _write_manifest(manifest_path, manifest)

    return CutlassStageInfo(
        path=stage_path,
        manifest_path=manifest_path,
        manifest=manifest,
        reused=False,
    )


def native_provenance(
    build_root: Path | str | None = None,
    *,
    prepare_cutlass: bool = False,
) -> dict[str, object]:
    """Return source provenance without compiling the native extension."""

    thirdparty = validate_thirdparty()
    provenance: dict[str, object] = {
        "root": str(ROOT),
        "thirdparty": {name: info.as_dict() for name, info in thirdparty.items()},
        "cutlass_patch": cutlass_patch_fingerprint(),
    }
    if prepare_cutlass:
        provenance["cutlass_stage"] = prepare_cutlass_stage(build_root).as_dict()
    return provenance
