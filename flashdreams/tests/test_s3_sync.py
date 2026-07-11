# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.core.io import s3_sync

pytestmark = pytest.mark.ci_cpu


def test_rank0_download_failure_is_broadcast_to_all_ranks(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingS3FileSystem:
        instances: list["_FailingS3FileSystem"] = []

        def __init__(self, credential_path: str) -> None:
            self.credential_path = credential_path
            self.closed = False
            self.instances.append(self)

        def list_files_recursive(self, s3_dir: str) -> list[str]:
            assert s3_dir == "s3://bucket/assets"
            return ["model.bin"]

        def download_to_local(self, s3_uri: str, local_path: str) -> None:
            assert s3_uri == "s3://bucket/assets/model.bin"
            raise OSError("S3 download failed")

        def close(self) -> None:
            self.closed = True

    current_rank = [0]
    transmitted_payload: list[dict[str, str | None]] = [{"error": None}]
    barriers: list[None] = []

    monkeypatch.setattr(s3_sync, "S3FileSystem", _FailingS3FileSystem)
    monkeypatch.setattr(s3_sync, "ensure_free_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(s3_sync, "cache_min_free_bytes", lambda: 0)
    monkeypatch.setattr(s3_sync.dist, "is_available", lambda: True)
    monkeypatch.setattr(s3_sync.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(s3_sync.dist, "get_rank", lambda: current_rank[0])
    monkeypatch.setattr(s3_sync.tqdm.tqdm, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(s3_sync, "_barrier_robust", lambda: barriers.append(None))

    def _broadcast_object_list(payload: list[dict[str, str | None]], src: int) -> None:
        assert src == 0
        if current_rank[0] == 0:
            transmitted_payload[0] = payload[0].copy()
        else:
            payload[0] = transmitted_payload[0].copy()

    monkeypatch.setattr(s3_sync.dist, "broadcast_object_list", _broadcast_object_list)

    for rank, expected_error in ((0, OSError), (1, RuntimeError)):
        current_rank[0] = rank
        with pytest.raises(expected_error, match="S3 download failed"):
            s3_sync.sync_s3_dir_to_local(
                s3_dir="s3://bucket/assets",
                s3_credential_path="credentials/s3.json",
                cache_dir=str(tmp_path),
                show_progress=False,
            )

    assert transmitted_payload == [{"error": "OSError: S3 download failed"}]
    assert len(_FailingS3FileSystem.instances) == 1
    assert _FailingS3FileSystem.instances[0].closed
    assert barriers == []


def test_single_process_failure_preserves_native_exception(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_s3_client(*, credential_path: str) -> None:
        assert credential_path == "credentials/s3.json"
        raise OSError("S3 client initialization failed")

    monkeypatch.setattr(s3_sync, "S3FileSystem", _fail_s3_client)
    monkeypatch.setattr(s3_sync, "ensure_free_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(s3_sync.dist, "is_available", lambda: False)

    with pytest.raises(OSError, match="S3 client initialization failed"):
        s3_sync.sync_s3_dir_to_local(
            s3_dir="s3://bucket/assets",
            s3_credential_path="credentials/s3.json",
            cache_dir=str(tmp_path),
        )
