# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno

import pytest

from flashdreams.core.io import hf as hf_io

pytestmark = pytest.mark.ci_cpu


def _clear_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("LOCAL_FILES_ONLY", raising=False)
    monkeypatch.setenv(hf_io.CACHE_MIN_FREE_ENV, "0")


def _mock_distributed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    initialized: bool,
    rank: int = 0,
    broadcast_error: str | None = None,
) -> list[list[dict[str, str | None]]]:
    broadcasts: list[list[dict[str, str | None]]] = []
    monkeypatch.setattr(hf_io, "is_distributed_initialized", lambda: initialized)
    monkeypatch.setattr(hf_io, "get_global_rank", lambda: rank if initialized else 0)

    def _broadcast_object_list(payload: list[dict[str, str | None]], src: int) -> None:
        assert src == 0
        if broadcast_error is not None:
            payload[0]["error"] = broadcast_error
        broadcasts.append(payload)

    monkeypatch.setattr(hf_io.dist, "broadcast_object_list", _broadcast_object_list)
    return broadcasts


def test_local_path_skips_snapshot_download(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    _mock_distributed(monkeypatch, initialized=False)

    def _fail_snapshot_download(*args, **kwargs) -> None:
        raise AssertionError("snapshot_download should not be called")

    monkeypatch.setattr(hf_io, "snapshot_download", _fail_snapshot_download)

    assert hf_io.maybe_download_hf_repo_on_rank0(str(tmp_path)) is None


def test_offline_env_skips_snapshot_download(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _mock_distributed(monkeypatch, initialized=False)

    def _fail_snapshot_download(*args, **kwargs) -> None:
        raise AssertionError("snapshot_download should not be called")

    monkeypatch.setattr(hf_io, "snapshot_download", _fail_snapshot_download)

    assert hf_io.maybe_download_hf_repo_on_rank0("org/model") is None


def test_remote_repo_downloads_requested_snapshot(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    _mock_distributed(monkeypatch, initialized=False)
    calls = []

    def _snapshot_download(*args, **kwargs) -> str:
        calls.append((args, kwargs))
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(hf_io, "snapshot_download", _snapshot_download)

    assert (
        hf_io.maybe_download_hf_repo_on_rank0(
            "org/model",
            revision="abc123",
            cache_dir=tmp_path,
            allow_patterns=("text_encoder/**", "tokenizer/**"),
        )
        is None
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("org/model",)
    assert str(kwargs.pop("cache_dir")) == str(tmp_path)
    assert kwargs == {
        "revision": "abc123",
        "local_files_only": False,
        "allow_patterns": ["text_encoder/**", "tokenizer/**"],
        "ignore_patterns": None,
    }


def test_remote_repo_uses_default_cache_dir(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    _mock_distributed(monkeypatch, initialized=False)
    monkeypatch.setattr(hf_io, "HUGGINGFACE_HUB_CACHE", str(tmp_path))
    calls = []

    def _snapshot_download(*args, **kwargs) -> str:
        calls.append((args, kwargs))
        return "/tmp/snapshot"

    monkeypatch.setattr(hf_io, "snapshot_download", _snapshot_download)

    assert hf_io.maybe_download_hf_repo_on_rank0("org/model") is None

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["cache_dir"] is None


def test_nonzero_distributed_rank_waits_for_rank0(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    broadcasts = _mock_distributed(monkeypatch, initialized=True, rank=1)

    def _fail_snapshot_download(*args, **kwargs) -> None:
        raise AssertionError("nonzero rank should not download")

    monkeypatch.setattr(hf_io, "snapshot_download", _fail_snapshot_download)

    assert (
        hf_io.maybe_download_hf_repo_on_rank0(
            "org/model",
            cache_dir=tmp_path,
            allow_patterns="image_encoder/**",
        )
        is None
    )
    assert broadcasts == [[{"error": None}]]


def test_rank0_download_failure_is_broadcast(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    broadcasts = _mock_distributed(monkeypatch, initialized=True, rank=0)

    def _snapshot_download(*args, **kwargs) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(hf_io, "snapshot_download", _snapshot_download)

    with pytest.raises(RuntimeError, match="OSError: disk full"):
        hf_io.maybe_download_hf_repo_on_rank0("org/model", cache_dir=tmp_path)

    assert broadcasts == [[{"error": "OSError: disk full"}]]


def test_rank0_disk_space_failure_is_broadcast_as_disk_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    broadcasts = _mock_distributed(monkeypatch, initialized=True, rank=0)

    def _snapshot_download(*args, **kwargs) -> str:
        raise OSError(errno.ENOSPC, "No space left on device", str(tmp_path))

    monkeypatch.setattr(hf_io, "snapshot_download", _snapshot_download)

    with pytest.raises(hf_io.DiskSpaceError, match="Hugging Face cache"):
        hf_io.maybe_download_hf_repo_on_rank0("org/model", cache_dir=tmp_path)

    assert len(broadcasts) == 1
    assert broadcasts[0][0]["disk_error"] == "1"
    assert "No space left on device" in str(broadcasts[0][0]["error"])


def test_rank0_failure_reaches_nonzero_rank(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_hf_env(monkeypatch)
    _mock_distributed(
        monkeypatch,
        initialized=True,
        rank=1,
        broadcast_error="OSError: disk full",
    )

    def _fail_snapshot_download(*args, **kwargs) -> None:
        raise AssertionError("nonzero rank should not download")

    monkeypatch.setattr(hf_io, "snapshot_download", _fail_snapshot_download)

    with pytest.raises(RuntimeError, match="OSError: disk full"):
        hf_io.maybe_download_hf_repo_on_rank0("org/model", cache_dir=tmp_path)
