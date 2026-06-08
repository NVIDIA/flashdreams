# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import errno
from types import SimpleNamespace

import pytest

from flashdreams.core.io import disk
from flashdreams.core.io.download import download_to_cache
from flashdreams.scripts import cli

pytestmark = pytest.mark.ci_cpu


def _fake_disk_usage(*, free: int) -> SimpleNamespace:
    return SimpleNamespace(total=10 * 1024**3, used=10 * 1024**3 - free, free=free)


def test_default_preflight_thresholds_are_runtime_reserves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(disk.CACHE_MIN_FREE_ENV, raising=False)
    monkeypatch.delenv(disk.OUTPUT_MIN_FREE_ENV, raising=False)
    monkeypatch.delenv(disk.TMP_MIN_FREE_ENV, raising=False)

    assert disk.cache_min_free_bytes() == disk.bytes_from_gib(20)
    assert disk.output_min_free_bytes() == disk.bytes_from_gib(20)
    assert disk.tmp_min_free_bytes() == disk.bytes_from_gib(20)


def test_model_cache_threshold_uses_model_default_unless_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(disk.CACHE_MIN_FREE_ENV, raising=False)
    assert disk.cache_min_free_bytes(default_gb=200) == disk.bytes_from_gib(200)

    monkeypatch.setenv(disk.CACHE_MIN_FREE_ENV, "5")
    assert disk.cache_min_free_bytes(default_gb=200) == disk.bytes_from_gib(5)


def test_huggingface_cache_dir_uses_hub_constant(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(disk, "HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "ignored"))

    assert disk.default_huggingface_cache_dir() == tmp_path / "hub"


def test_ensure_free_disk_reports_path_free_required_and_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        disk.shutil,
        "disk_usage",
        lambda path: _fake_disk_usage(free=512 * 1024**2),
    )

    with pytest.raises(disk.DiskSpaceError) as exc_info:
        disk.ensure_free_disk(
            tmp_path / "cache",
            required_bytes=disk.bytes_from_gib(2),
            label="FlashDreams cache",
            env_vars=("FLASHDREAMS_CACHE_DIR", disk.CACHE_MIN_FREE_ENV),
        )

    message = str(exc_info.value)
    assert "ERROR: Not enough free disk for FlashDreams cache." in message
    assert f"Path:     {tmp_path / 'cache'}" in message
    assert "Free:     512.0 MiB" in message
    assert "Required: 2.0 GiB" in message
    assert "FLASHDREAMS_CACHE_DIR:" in message
    assert f"{disk.CACHE_MIN_FREE_ENV}:" in message
    assert "Move FLASHDREAMS_CACHE_DIR to a filesystem with more free space." in message


def test_enospc_exception_chain_formats_cause_path_and_settings(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "out.mp4"
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setattr(
        disk.shutil,
        "disk_usage",
        lambda path: _fake_disk_usage(free=0),
    )

    try:
        raise RuntimeError("video writer failed") from OSError(
            errno.ENOSPC,
            "No space left on device",
            str(output_path),
        )
    except RuntimeError as exc:
        disk_error = disk.disk_space_error_from_exception(
            exc,
            label="output video",
            required_bytes=disk.bytes_from_gib(1),
            env_vars=("TMPDIR", disk.OUTPUT_MIN_FREE_ENV),
            settings={"--output-dir": tmp_path},
        )

    assert disk_error is not None
    message = str(disk_error)
    assert "ERROR: Disk space exhausted while writing output video." in message
    assert f"Path:     {output_path}" in message
    assert "Free:     0 B" in message
    assert "Required: 1.0 GiB" in message
    assert "Cause:    OSError: [Errno 28] No space left on device" in message
    assert "--output-dir:" in message
    assert "TMPDIR:" in message
    assert "Move TMPDIR, --output-dir to a filesystem with more free space." in message


def test_suppressed_enospc_context_is_not_treated_as_disk_error(
    tmp_path,
) -> None:
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device", str(tmp_path))
        except OSError:
            raise RuntimeError("real non-disk failure") from None
    except RuntimeError as exc:
        assert disk.disk_space_error_from_exception(exc) is None


def test_cli_disk_handler_prints_message_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _raise_disk_error() -> None:
        raise disk.DiskSpaceError("ERROR: disk is full\n  Path: /tmp/out")

    with pytest.raises(SystemExit) as exc_info:
        cli._run_with_disk_error_handling(_raise_disk_error)

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "ERROR: disk is full" in stderr
    assert "Traceback" not in stderr


def test_download_to_cache_converts_enospc(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setenv(disk.CACHE_MIN_FREE_ENV, "0")
    monkeypatch.setattr(
        "flashdreams.core.io.download.urllib.request.urlopen",
        lambda *args, **kwargs: _Response(),
    )
    monkeypatch.setattr(
        "flashdreams.core.io.download.shutil.copyfileobj",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.ENOSPC, "No space left on device")
        ),
    )

    with pytest.raises(disk.DiskSpaceError) as exc_info:
        download_to_cache("https://example.test/frame.png", cache_dir=tmp_path)

    message = str(exc_info.value)
    assert (
        "ERROR: Disk space exhausted while writing FlashDreams cache download."
        in message
    )
    assert "frame.png" in message
    assert "https://example.test/frame.png" in message
