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

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpadreams.native import omnidreams_singleview as native


_FAKE_THIRDPARTY_INFO = {
    "cutlass": {
        "name": "cutlass",
        "path": "integrations/alpadreams/omnidreams_singleview/3rdparty/cutlass",
        "recorded_sha": "cutlass-test-sha",
        "head_sha": "cutlass-test-sha",
        "tracked_status": [],
    },
    "SageAttention": {
        "name": "SageAttention",
        "path": "integrations/alpadreams/omnidreams_singleview/3rdparty/SageAttention",
        "recorded_sha": "sage-test-sha",
        "head_sha": "sage-test-sha",
        "tracked_status": [],
    },
    "SpargeAttn": {
        "name": "SpargeAttn",
        "path": "integrations/alpadreams/omnidreams_singleview/3rdparty/SpargeAttn",
        "recorded_sha": "sparge-test-sha",
        "head_sha": "sparge-test-sha",
        "tracked_status": [],
    },
}


def _run_git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip()


@pytest.mark.ci_cpu
def test_thirdparty_submodules_match_parent_gitlinks() -> None:
    info = native.validate_thirdparty()

    assert set(info) == {"cutlass", "SageAttention", "SpargeAttn"}
    for submodule_info in info.values():
        assert submodule_info["head_sha"] == submodule_info["recorded_sha"]
        assert submodule_info["tracked_status"] == []


@pytest.mark.ci_cpu
def test_cutlass_patch_applies_to_pinned_submodule() -> None:
    helper = native._native_build()
    _run_git(helper.CUTLASS_DIR, "apply", "--check", str(helper.CUTLASS_PATCH))


@pytest.mark.ci_cpu
def test_cutlass_stage_refreshes_and_reuses_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = native._native_build()

    fake_cutlass = tmp_path / "cutlass"
    fake_cutlass.mkdir()
    _run_git(tmp_path, "init", str(fake_cutlass))
    (fake_cutlass / "include" / "cute").mkdir(parents=True)
    (fake_cutlass / "include" / "cute" / "example.hpp").write_text(
        "original\n",
        encoding="utf-8",
    )
    _run_git(fake_cutlass, "add", ".")
    _run_git(
        fake_cutlass,
        "-c",
        "user.name=FlashDreams Test",
        "-c",
        "user.email=flashdreams-test@nvidia.com",
        "commit",
        "-m",
        "seed fake cutlass",
    )

    patch = tmp_path / "patch.diff"
    patch.write_text(
        """diff --git a/include/cute/example.hpp b/include/cute/example.hpp
--- a/include/cute/example.hpp
+++ b/include/cute/example.hpp
@@ -1 +1,2 @@
 original
+patched
""",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay"
    (overlay / "cute" / "atom" / "detail").mkdir(parents=True)
    (overlay / "cute" / "atom" / "detail" / "extra.hpp").write_text(
        "overlay\n",
        encoding="utf-8",
    )

    fake_info = helper.SubmoduleInfo(
        name="cutlass",
        path=fake_cutlass,
        relative_path="fake/cutlass",
        recorded_sha="fake-sha",
        head_sha="fake-sha",
        tracked_status=(),
    )
    monkeypatch.setattr(helper, "CUTLASS_DIR", fake_cutlass)
    monkeypatch.setattr(helper, "CUTLASS_PATCH", patch)
    monkeypatch.setattr(helper, "CUTLASS_OVERLAY_INCLUDE", overlay)
    monkeypatch.setattr(
        helper,
        "validate_thirdparty",
        lambda: {
            "cutlass": fake_info,
            "SageAttention": fake_info,
            "SpargeAttn": fake_info,
        },
    )

    first_stage = helper.prepare_cutlass_stage(tmp_path / "build")
    assert first_stage.reused is False
    assert (
        first_stage.path / "include" / "cute" / "example.hpp"
    ).read_text(encoding="utf-8") == "original\npatched\n"
    assert (
        first_stage.path / "include" / "cute" / "atom" / "detail" / "extra.hpp"
    ).exists()
    assert first_stage.manifest["cutlass_sha"] == "fake-sha"

    second_stage = helper.prepare_cutlass_stage(tmp_path / "build")
    assert second_stage.reused is True
    assert _run_git(fake_cutlass, "status", "--short", "--untracked-files=no") == ""


@pytest.mark.ci_cpu
def test_load_extension_uses_build_root_for_torch_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    build_root = tmp_path / "native-build"
    stage_path = build_root / "_deps" / "cutlass-patched"
    (stage_path / "include").mkdir(parents=True)

    stage_info = {
        "path": str(stage_path),
        "manifest_path": str(stage_path / ".omnidreams_cutlass_stage.json"),
        "manifest": {
            "cutlass_sha": "cutlass-test-sha",
            "patch_sha256": "patch-test-sha",
            "overlay_include_sha256": "overlay-test-sha",
        },
        "reused": False,
    }
    captured: dict[str, object] = {}

    def fake_load_torch_extension(**kwargs: object) -> object:
        captured.update(kwargs)
        captured["max_jobs_env"] = os.environ.get("MAX_JOBS")
        return SimpleNamespace()

    monkeypatch.setattr(native, "_extension", None)
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "prepare_cutlass_stage", lambda **_: stage_info)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: _FAKE_THIRDPARTY_INFO)
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.delenv("OMNIDREAMS_SINGLEVIEW_NATIVE_MAX_JOBS", raising=False)

    extension = native.load_extension(build_root=build_root)

    assert extension is not None
    extension_name = captured["name"]
    assert captured["build_directory"] == str(
        build_root / "torch_extensions" / str(extension_name)
    )
    assert captured["extra_include_paths"] == [str(stage_path / "include")]
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SHA=\\\"cutlass-test-sha\\\""
        in captured["extra_cflags"]
    )
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_PATCH_SHA=\\\"patch-test-sha\\\""
        in captured["extra_cflags"]
    )
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_OVERLAY_SHA=\\\"overlay-test-sha\\\""
        in captured["extra_cflags"]
    )
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_SAGE_ATTENTION_SHA=\\\"sage-test-sha\\\""
        in captured["extra_cflags"]
    )
    assert (
        "-DOMNIDREAMS_SINGLEVIEW_SPARGE_ATTN_SHA=\\\"sparge-test-sha\\\""
        in captured["extra_cflags"]
    )
    assert captured["with_cuda"] is False
    assert captured["max_jobs_env"] == "1"
    assert "MAX_JOBS" not in os.environ


@pytest.mark.ci_cpu
def test_load_extension_respects_existing_max_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    build_root = tmp_path / "native-build"
    stage_path = build_root / "_deps" / "cutlass-patched"
    (stage_path / "include").mkdir(parents=True)

    stage_info = {
        "path": str(stage_path),
        "manifest_path": str(stage_path / ".omnidreams_cutlass_stage.json"),
        "manifest": {
            "cutlass_sha": "cutlass-test-sha",
            "patch_sha256": "patch-test-sha",
            "overlay_include_sha256": "overlay-test-sha",
        },
        "reused": False,
    }
    captured: dict[str, object] = {}

    def fake_load_torch_extension(**kwargs: object) -> object:
        captured["max_jobs_env"] = os.environ.get("MAX_JOBS")
        return SimpleNamespace()

    monkeypatch.setattr(native, "_extension", None)
    monkeypatch.setattr(native, "_extension_load_error", None)
    monkeypatch.setattr(native, "prepare_cutlass_stage", lambda **_: stage_info)
    monkeypatch.setattr(native, "validate_thirdparty", lambda: _FAKE_THIRDPARTY_INFO)
    monkeypatch.setattr(cpp_extension, "load", fake_load_torch_extension)
    monkeypatch.setenv("MAX_JOBS", "3")

    extension = native.load_extension(build_root=build_root)

    assert extension is not None
    assert captured["max_jobs_env"] == "3"
    assert os.environ["MAX_JOBS"] == "3"


@pytest.mark.ci_cpu
@pytest.mark.skipif(
    os.environ.get("OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST") != "1",
    reason="Set OMNIDREAMS_SINGLEVIEW_RUN_NATIVE_BUILD_TEST=1 to build the native extension.",
)
def test_diagnostic_native_extension_builds(tmp_path: Path) -> None:
    extension = native.load_extension(build_root=tmp_path)

    assert extension is not None, native.extension_load_error()
    assert extension.is_available()
    assert extension.build_info()["with_cuda"] is False
