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

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.ci_cpu

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "omnidreams_singleview"
    / "tools"
    / "sync_thirdparty.py"
)


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_thirdparty", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    _run_git(repo, "init", "-q")
    (repo / "hello.txt").write_text("old\n", encoding="utf-8")
    (repo / "remove_me").mkdir()
    (repo / "remove_me" / "unused.txt").write_text("unused\n", encoding="utf-8")
    _run_git(repo, "add", ".")
    _run_git(
        repo,
        "-c",
        "user.name=FlashDreams Test",
        "-c",
        "user.email=flashdreams-test@nvidia.com",
        "commit",
        "-q",
        "-m",
        "initial",
    )
    return repo, _run_git(repo, "rev-parse", "HEAD")


def _write_manifest(
    path: Path,
    *,
    repo: Path,
    commit: str,
    patch: Path | None = None,
) -> None:
    source: dict[str, Any] = {
        "name": "demo",
        "repo": str(repo),
        "commit": commit,
        "directory": "demo",
        "delete_paths": ["remove_me"],
    }
    if patch is not None:
        source["patches"] = [{"path": str(patch), "strip": 1}]

    path.write_text(
        json.dumps({"schema_version": 1, "sources": [source]}, indent=2) + "\n",
        encoding="utf-8",
    )


def test_sync_downloads_pinned_source_and_applies_operations(tmp_path: Path) -> None:
    module = _load_sync_module()
    repo, commit = _make_repo(tmp_path)
    patch = tmp_path / "hello.patch"
    patch.write_text(
        """diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-old
+new
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "thirdparty_sources.json"
    dest_root = tmp_path / "3rdparty"
    _write_manifest(manifest, repo=repo, commit=commit, patch=patch)

    sources = module.load_manifest(manifest)
    results = module.sync_sources(sources, dest_root)

    assert [result.source.name for result in results] == ["demo"]
    assert (dest_root / "demo" / "hello.txt").read_text(encoding="utf-8") == "new\n"
    assert not (dest_root / "demo" / "remove_me").exists()
    assert (dest_root / "demo" / ".flashdreams_source.json").is_file()
    assert module.verify_sources(sources, dest_root)[0].commit == commit

    (dest_root / "demo" / "hello.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(module.ThirdPartySyncError, match="source tree does not match"):
        module.verify_sources(sources, dest_root)


def test_sync_refuses_unmanaged_existing_directory(tmp_path: Path) -> None:
    module = _load_sync_module()
    repo, commit = _make_repo(tmp_path)
    manifest = tmp_path / "thirdparty_sources.json"
    dest_root = tmp_path / "3rdparty"
    _write_manifest(manifest, repo=repo, commit=commit)
    (dest_root / "demo").mkdir(parents=True)

    with pytest.raises(module.ThirdPartySyncError, match="not a Git checkout"):
        module.sync_sources(module.load_manifest(manifest), dest_root)


def test_empty_manifest_is_valid(tmp_path: Path) -> None:
    module = _load_sync_module()
    manifest = tmp_path / "thirdparty_sources.json"
    manifest.write_text('{"schema_version": 1, "sources": []}\n', encoding="utf-8")

    assert module.load_manifest(manifest) == ()


def test_manifest_read_errors_are_sync_errors(tmp_path: Path) -> None:
    module = _load_sync_module()

    with pytest.raises(module.ThirdPartySyncError, match="Manifest does not exist"):
        module.load_manifest(tmp_path / "missing.json")

    manifest = tmp_path / "thirdparty_sources.json"
    manifest.write_text("{not json\n", encoding="utf-8")

    with pytest.raises(module.ThirdPartySyncError, match="Invalid manifest JSON"):
        module.load_manifest(manifest)


def test_verify_reports_missing_management_stamp(tmp_path: Path) -> None:
    module = _load_sync_module()
    repo, commit = _make_repo(tmp_path)
    manifest = tmp_path / "thirdparty_sources.json"
    dest_root = tmp_path / "3rdparty"
    _write_manifest(manifest, repo=repo, commit=commit)

    sources = module.load_manifest(manifest)
    module.sync_sources(sources, dest_root)
    (dest_root / "demo" / ".flashdreams_source.json").unlink()

    with pytest.raises(
        module.ThirdPartySyncError, match="missing .flashdreams_source.json"
    ):
        module.verify_sources(sources, dest_root)
