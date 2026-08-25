# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for local and Hugging Face LingBot checkpoint resolution."""

from pathlib import Path
from typing import Any

import huggingface_hub
import pytest

import lingbot_va._loaders as loaders

pytestmark = pytest.mark.ci_cpu

_COMPONENTS = ("transformer", "vae", "text_encoder", "tokenizer")


def _checkpoint_root(path: Path) -> Path:
    path.mkdir()
    for component in _COMPONENTS:
        (path / component).mkdir()
    return path


def test_validate_checkpoint_root_accepts_complete_local_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint_root = _checkpoint_root(tmp_path / "checkpoint")

    assert loaders.validate_checkpoint_root(checkpoint_root) == checkpoint_root


def test_validate_checkpoint_root_names_missing_component(tmp_path: Path) -> None:
    checkpoint_root = _checkpoint_root(tmp_path / "checkpoint")
    (checkpoint_root / "tokenizer").rmdir()

    with pytest.raises(FileNotFoundError, match="tokenizer"):
        loaders.validate_checkpoint_root(checkpoint_root)


def test_resolve_local_checkpoint_never_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = _checkpoint_root(tmp_path / "checkpoint")

    def unexpected_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(loaders, "maybe_download_hf_repo_on_rank0", unexpected_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_download)

    assert loaders.resolve_checkpoint_root(checkpoint_root) == checkpoint_root


def test_resolve_remote_checkpoint_propagates_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = _checkpoint_root(tmp_path / "snapshot")
    calls: list[tuple[str, dict[str, Any]]] = []

    def record_preload(repo_id: str, **kwargs: Any) -> None:
        calls.append((repo_id, kwargs))

    def record_snapshot(**kwargs: Any) -> str:
        calls.append(("snapshot_download", kwargs))
        return str(checkpoint_root)

    monkeypatch.setattr(loaders, "maybe_download_hf_repo_on_rank0", record_preload)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", record_snapshot)

    resolved = loaders.resolve_checkpoint_root("owner/repo", revision="deadbeef")

    assert resolved == checkpoint_root
    assert calls[0][0] == "owner/repo"
    assert calls[0][1]["revision"] == "deadbeef"
    assert calls[1][1]["repo_id"] == "owner/repo"
    assert calls[1][1]["revision"] == "deadbeef"
    assert calls[1][1]["local_files_only"] is True


def test_resolve_nonexistent_explicit_path_fails_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing"

    def unexpected_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(loaders, "maybe_download_hf_repo_on_rank0", unexpected_download)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        loaders.resolve_checkpoint_root(missing)
