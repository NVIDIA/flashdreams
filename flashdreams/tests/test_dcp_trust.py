# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for DCP provenance and verified snapshot loading."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

import flashdreams.core.io.dcp_trust as dcp_trust
from flashdreams.core.checkpoint.load import (
    _s3_cache_path,
    get_storage_reader,
    load_checkpoint,
    load_distributed_checkpoint,
)
from flashdreams.core.io.dcp_trust import (
    DCPTrust,
    DCPTrustError,
    VerifiedDCPReader,
    main,
    resolve_dcp_checkpoint_path,
)

pytestmark = pytest.mark.ci_cpu


@pytest.fixture(autouse=True)
def _isolate_trust_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLASHDREAMS_DCP_TRUST_STORE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-default-store"))


def _pin(tmp_path: Path, files: dict[str, bytes], **changes: Any) -> DCPTrust:
    manifest = {
        "version": 1,
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for name, data in files.items()
        },
    }
    manifest.update(changes)
    return _pin_bytes(tmp_path, json.dumps(manifest).encode())


def _pin_bytes(tmp_path: Path, data: bytes) -> DCPTrust:
    manifest_path = tmp_path / "trusted-manifest.json"
    manifest_path.write_bytes(data)
    return DCPTrust(str(manifest_path), hashlib.sha256(data).hexdigest())


def _source(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    source = tmp_path / "source"
    source.mkdir()
    files = {".metadata": b"approved metadata", "__0_0.distcp": b"approved shard"}
    for name, data in files.items():
        (source / name).write_bytes(data)
    return source, files


def _tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("unverified bytes reached the DCP metadata parser")

    monkeypatch.setattr(FileSystemReader, "read_metadata", reject)


def _store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    trust: DCPTrust,
) -> None:
    store = tmp_path / "trust-store.json"
    store.write_text(
        json.dumps(
            {
                source: {
                    "manifest_path": str(trust.manifest_path),
                    "sha256": trust.sha256,
                }
            }
        )
    )
    store.chmod(0o600)
    monkeypatch.setenv("FLASHDREAMS_DCP_TRUST_STORE", str(store))


@pytest.mark.parametrize("entrypoint", [VerifiedDCPReader, get_storage_reader])
def test_unapproved_source_stops_before_metadata_parser(
    entrypoint: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source(tmp_path)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        entrypoint(str(source)).read_metadata()


@pytest.mark.parametrize("name", [".metadata", "__0_0.distcp"])
@pytest.mark.parametrize("replacement", ["digest", "size", "missing"])
def test_replaced_or_missing_files_stop_before_metadata_parser(
    name: str,
    replacement: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    if replacement == "missing":
        (source / name).unlink()
    else:
        (source / name).write_bytes(
            b"x" * len(files[name]) if replacement == "digest" else b"modified"
        )
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        VerifiedDCPReader(str(source), trust=trust).read_metadata()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
@pytest.mark.parametrize("replaced", [".metadata", "manifest"])
def test_fifo_in_place_of_pinned_file_fails_without_blocking(
    replaced: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    fifo = Path(trust.manifest_path) if replaced == "manifest" else source / replaced
    fifo.unlink()
    os.mkfifo(fifo)
    _tripwire(monkeypatch)
    errors: list[BaseException] = []

    def construct() -> None:
        try:
            VerifiedDCPReader(str(source), trust=trust).read_metadata()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=construct, daemon=True)
    worker.start()
    worker.join(timeout=5)
    if worker.is_alive():
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))  # release the reader
        pytest.fail(f"a FIFO at {replaced} blocked verified reader construction")
    assert len(errors) == 1
    assert isinstance(errors[0], DCPTrustError), errors[0]
    assert "not a regular file" in str(errors[0])


@pytest.mark.parametrize("via", ["explicit", "trust-store"])
def test_well_formed_substituted_manifest_fails_original_pin(
    via: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    replaced = {name: b"substituted " + data for name, data in files.items()}
    for name, data in replaced.items():
        (source / name).write_bytes(data)
    assert _pin(tmp_path, replaced).manifest_path == trust.manifest_path
    if via == "trust-store":
        _store(tmp_path, monkeypatch, str(source), trust)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError, match="manifest SHA-256 mismatch"):
        get_storage_reader(
            str(source), dcp_trust=trust if via == "explicit" else None
        ).read_metadata()


def test_trust_errors_name_the_source_record_and_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import re

    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    _store(tmp_path, monkeypatch, str(tmp_path / "other"), trust)
    with pytest.raises(DCPTrustError, match=re.escape(f"{source} is not listed")):
        get_storage_reader(str(source))
    _store(tmp_path, monkeypatch, str(source), DCPTrust("relative", trust.sha256))
    with pytest.raises(DCPTrustError, match=re.escape(f"record for {source}:")):
        get_storage_reader(str(source))
    (source / "__0_0.distcp").write_bytes(b"x" * len(files["__0_0.distcp"]))
    with pytest.raises(DCPTrustError, match=r"'__0_0\.distcp'.*expected 14 bytes"):
        VerifiedDCPReader(str(source), trust=trust)


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
@pytest.mark.parametrize("mode", [0o620, 0o602], ids=["group", "other"])
def test_shared_writable_trust_store_is_rejected(
    mode: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    _store(tmp_path, monkeypatch, str(source), _pin(tmp_path, files))
    (tmp_path / "trust-store.json").chmod(mode)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError, match="group- or world-writable.*chmod 600"):
        get_storage_reader(str(source)).read_metadata()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
@pytest.mark.parametrize("nested", [False, True], ids=["parent", "ancestor"])
def test_trust_store_in_shared_writable_directory_is_rejected(
    nested: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    shared = tmp_path / "shared"
    # Swapping a private folder inside a shared one also swaps the store.
    folder = shared / "config" if nested else shared
    folder.mkdir(parents=True)
    store = folder / "trust-store.json"
    store.write_text(
        json.dumps(
            {
                str(source): {
                    "manifest_path": trust.manifest_path,
                    "sha256": trust.sha256,
                }
            }
        )
    )
    store.chmod(0o644)  # not writable by others, but its directory is
    monkeypatch.setenv("FLASHDREAMS_DCP_TRUST_STORE", str(store))
    shared.chmod(0o777)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError, match=f"chmod go-w {shared}"):
        get_storage_reader(str(source))
    shared.chmod(0o1777)  # the sticky bit stops others replacing the store
    assert resolve_dcp_checkpoint_path(str(source)) == str(source)


@pytest.mark.skipif(not dcp_trust._DIR_FD, reason="needs directory descriptors")
def test_renamed_snapshot_directory_cannot_swap_parsed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    scratch = tmp_path / "shared-scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    source = tmp_path / "source"
    dcp.save({"approved": torch.ones(2)}, checkpoint_id=source)
    files = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    trust = _pin(tmp_path, files)
    replacement = tmp_path / "replacement"
    dcp.save({"planted": torch.zeros(2)}, checkpoint_id=replacement)

    with VerifiedDCPReader(str(source), trust=trust) as reader:
        # Another account with write access to the scratch parent swaps names.
        snapshot = Path(reader.path)
        moved = scratch / "moved-away"
        snapshot.rename(moved)
        replacement.rename(snapshot)
        assert set(reader.read_metadata().state_dict_metadata) == {"approved"}
        actual = {"approved": torch.zeros(2)}
        dcp.load(actual, storage_reader=reader)
    torch.testing.assert_close(actual["approved"], torch.ones(2))
    # Closing deletes the verified copy where it now is, not what has its name.
    assert not any(moved.iterdir())
    assert (snapshot / ".metadata").is_file()


def test_snapshot_without_directory_descriptors_removes_read_only_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows has no ``dir_fd`` and refuses to delete read-only files."""
    monkeypatch.setattr(dcp_trust, "_DIR_FD", False)
    unlink = os.unlink

    def windows_unlink(path: Any, *, dir_fd: int | None = None) -> None:
        if not os.stat(path, dir_fd=dir_fd).st_mode & stat.S_IWUSR:
            raise PermissionError(path)
        unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", windows_unlink)
    source, files = _source(tmp_path)
    with VerifiedDCPReader(str(source), trust=_pin(tmp_path, files)) as reader:
        snapshot = Path(reader.path)
        assert (snapshot / "__0_0.distcp").is_file()
    assert not snapshot.exists()


def test_unclosed_reader_snapshot_is_removed_when_collected(tmp_path: Path) -> None:
    source, files = _source(tmp_path)
    reader = VerifiedDCPReader(str(source), trust=_pin(tmp_path, files))
    snapshot = Path(reader.path)
    assert snapshot.is_dir()
    del reader
    gc.collect()
    assert not snapshot.exists()


def test_pin_command_approves_without_setup_and_checks_publisher_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "producer"))
    source = tmp_path / "export" / "model"
    expected = torch.arange(4, dtype=torch.float32)
    dcp.save({"weight": expected}, checkpoint_id=source)
    assert main(["pin", str(source)]) == 0
    store = tmp_path / "producer" / "flashdreams" / "dcp-trust.json"
    if os.name != "nt":
        assert stat.S_IMODE(store.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.parent.stat().st_mode) == 0o700
    with get_storage_reader(str(source)) as reader:
        actual = {"weight": torch.zeros_like(expected)}
        dcp.load(actual, storage_reader=reader)
    torch.testing.assert_close(actual["weight"], expected)

    # A consumer registers the publisher's manifest only with a matching digest.
    pin = json.loads(store.read_text())[str(source)]
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "consumer"))
    remote = "s3://bucket/run/model/"
    with pytest.raises(SystemExit):
        main(["pin", "--manifest", pin["manifest_path"], "--sha256", "0" * 64, remote])
    assert "does not match --sha256" in capsys.readouterr().err
    assert not (tmp_path / "consumer" / "flashdreams" / "dcp-trust.json").exists()
    args = ["pin", "--manifest", pin["manifest_path"], "--sha256", pin["sha256"]]
    assert main([*args, remote]) == 0
    assert resolve_dcp_checkpoint_path(remote) == remote.rstrip("/")


def test_full_snapshot_disk_is_not_reported_as_trust_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    from flashdreams.core.io.disk import DiskSpaceError

    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)

    class FullDisk(io.BytesIO):
        def write(self, _data: Any) -> int:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(dcp_trust, "_create_snapshot_file", lambda *_: FullDisk())
    _tripwire(monkeypatch)
    with pytest.raises(DiskSpaceError, match="verified DCP snapshot") as failure:
        VerifiedDCPReader(str(source), trust=trust)
    assert not isinstance(failure.value, DCPTrustError)


def test_changed_manifest_stops_before_metadata_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    Path(trust.manifest_path).write_text('{"version":1,"files":{}}')
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        VerifiedDCPReader(str(source), trust=trust).read_metadata()


@pytest.mark.parametrize(
    "manifest",
    [
        b"not JSON",
        b"[]",
        b'{"version":1,"version":1,"files":{}}',
        b'{"version":2,"files":{}}',
        b'{"version":1,"files":{}}',
        b'{"version":1,"files":[],"unexpected":true}',
        b'{"version":1,"files":{".metadata":{"sha256":"bad","size":1}}}',
        b'{"version":1,"files":{".metadata":{"sha256":"'
        + b"a" * 64
        + b'","size":-1}}}',
        b'{"version":1,"files":{".metadata":{"sha256":"'
        + b"a" * 64
        + b'","size":true}}}',
    ],
)
def test_pinned_malformed_manifest_stops_before_metadata_parser(
    manifest: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source(tmp_path)
    trust = _pin_bytes(tmp_path, manifest)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        VerifiedDCPReader(str(source), trust=trust).read_metadata()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../../outside"])
def test_manifest_cannot_escape_snapshot(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    files[name] = b"outside"
    trust = _pin(tmp_path, files)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        VerifiedDCPReader(str(source), trust=trust).read_metadata()


def test_real_local_dcp_metadata_and_weights_load_with_offline_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dcp.save({"weight": expected}, checkpoint_id=source)
    files = {
        path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
    }
    trust = _pin(tmp_path, files)
    _store(tmp_path, monkeypatch, str(source.resolve()), trust)

    with get_storage_reader(str(source)) as reader:
        metadata = reader.read_metadata()
        assert "weight" in metadata.state_dict_metadata
        actual = {"weight": torch.zeros_like(expected)}
        dcp.load(actual, storage_reader=reader)
        torch.testing.assert_close(actual["weight"], expected)

    with get_storage_reader(str(source), dcp_trust=trust) as reader:
        assert "weight" in reader.read_metadata().state_dict_metadata

    model = torch.nn.Linear(4, 3, bias=False)
    assert (
        load_checkpoint(str(source), model, dcp_trust=trust, check_success=True)
        is model
    )
    torch.testing.assert_close(model.weight, expected)


@pytest.mark.parametrize("failure", ["changed-manifest", "unlisted-source"])
def test_trust_store_cannot_approve_substituted_or_unlisted_input(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    key = source if failure == "changed-manifest" else tmp_path / "other-source"
    _store(tmp_path, monkeypatch, str(key.resolve()), trust)
    if failure == "changed-manifest":
        Path(trust.manifest_path).write_text('{"version":1,"files":{}}')
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        get_storage_reader(str(source)).read_metadata()


@pytest.mark.parametrize("approved", ["root", "model", "both"])
def test_resolver_selects_only_pinned_root_or_model_directory(
    approved: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    pin = {"manifest_path": str(trust.manifest_path), "sha256": trust.sha256}
    entries = {}
    if approved != "model":
        entries[str(source)] = pin
    if approved != "root":
        entries[str(source / "model")] = pin
    store = tmp_path / "trust-store.json"
    store.write_text(json.dumps(entries))
    store.chmod(0o600)
    monkeypatch.setenv("FLASHDREAMS_DCP_TRUST_STORE", str(store))

    expected = source / "model" if approved == "model" else source
    assert resolve_dcp_checkpoint_path(str(source)) == str(expected)


def test_resolver_does_not_fallback_after_selected_root_fails_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    pin = {"manifest_path": str(trust.manifest_path), "sha256": trust.sha256}
    model = source / "model"
    model.mkdir()
    for name, data in files.items():
        (model / name).write_bytes(data)
    (source / ".metadata").write_bytes(b"modified")
    store = tmp_path / "trust-store.json"
    store.write_text(json.dumps({str(source): pin, str(model): pin}))
    store.chmod(0o600)
    monkeypatch.setenv("FLASHDREAMS_DCP_TRUST_STORE", str(store))
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        get_storage_reader(resolve_dcp_checkpoint_path(str(source))).read_metadata()


def test_failed_verification_removes_partial_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    (source / "__0_0.distcp").write_bytes(b"modified")
    make_directory = tempfile.mkdtemp
    snapshots = []

    def record_snapshot(*args: Any, **kwargs: Any) -> Any:
        snapshot = make_directory(*args, **kwargs)
        snapshots.append(Path(snapshot))
        return snapshot

    monkeypatch.setattr(tempfile, "mkdtemp", record_snapshot)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        VerifiedDCPReader(str(source), trust=trust).read_metadata()
    assert snapshots
    assert all(not snapshot.exists() for snapshot in snapshots)


def test_unlisted_shard_cannot_be_read_after_approved_metadata(
    tmp_path: Path,
) -> None:
    from torch.distributed.checkpoint.api import CheckpointException

    source = tmp_path / "source"
    expected = torch.arange(3, dtype=torch.float32)
    dcp.save({"weight": expected}, checkpoint_id=source)
    trust = _pin(tmp_path, {".metadata": (source / ".metadata").read_bytes()})
    actual = {"weight": torch.zeros_like(expected)}
    with VerifiedDCPReader(str(source), trust=trust) as reader:
        with pytest.raises(CheckpointException, match="not covered"):
            dcp.load(actual, storage_reader=reader)
    torch.testing.assert_close(actual["weight"], torch.zeros_like(expected))


def test_parser_uses_private_snapshot_after_source_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)

    def inspect_snapshot(
        reader: FileSystemReader, *_args: Any, **_kwargs: Any
    ) -> bytes:
        snapshot = Path(reader.path)
        assert snapshot != source
        if os.name != "nt":
            assert stat.S_IMODE(snapshot.stat().st_mode) & 0o077 == 0
        assert (snapshot / "__0_0.distcp").read_bytes() == files["__0_0.distcp"]
        return (snapshot / ".metadata").read_bytes()

    monkeypatch.setattr(FileSystemReader, "read_metadata", inspect_snapshot)
    with VerifiedDCPReader(str(source), trust=trust) as reader:
        snapshot = Path(reader.path)
        for name in files:
            replacement = source / "replacement"
            replacement.write_bytes(b"replacement")
            replacement.replace(source / name)
        assert reader.read_metadata() == files[".metadata"]
        reader.reset(str(source))
        assert reader.read_metadata() == files[".metadata"]
        with pytest.raises(DCPTrustError):
            reader.reset(str(tmp_path / "unchecked"))
    assert not snapshot.exists()


def test_mutation_during_copy_stops_before_metadata_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    import flashdreams.core.io.dcp_trust as dcp_trust

    source, files = _source(tmp_path)
    trust = _pin(tmp_path, files)
    original_open = dcp_trust._open_regular

    class RacingStream:
        def __init__(self, stream: Any) -> None:
            self.stream = stream
            self.started = False
            self.remaining = io.BytesIO()

        def read(self, size: int = -1) -> bytes:
            if not self.started:
                self.started = True
                prefix = self.stream.read(4)
                changed = b"xxxx" + b"!" * (len(files[".metadata"]) - 4)
                (source / ".metadata").write_bytes(changed)
                self.remaining = io.BytesIO(changed[4:])
                return prefix
            return self.remaining.read(size)

    @contextmanager
    def racing_stream(path: Any) -> Any:
        with original_open(path) as stream:
            if Path(path) == source / ".metadata":
                yield RacingStream(stream)
            else:
                yield stream

    monkeypatch.setattr(dcp_trust, "_open_regular", racing_stream)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        VerifiedDCPReader(str(source), trust=trust).read_metadata()


def test_unlisted_rank_metadata_cannot_reach_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, files = _source(tmp_path)
    (source / "__1.metadata").write_bytes(b"unapproved rank metadata")
    trust = _pin(tmp_path, files)
    with VerifiedDCPReader(str(source), trust=trust) as reader:
        _tripwire(monkeypatch)
        with pytest.raises(DCPTrustError):
            reader.read_metadata(rank=1)


@pytest.mark.parametrize("approved_source", [False, True])
def test_merged_s3_cache_cannot_bypass_trust_or_substituted_source(
    approved_source: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = "s3://bucket/model"
    cache_path = Path(_s3_cache_path(str(tmp_path), source, suffix=".pt"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": torch.ones(1, 1)}, cache_path)
    files = {".metadata": b"approved metadata", "__0_0.distcp": b"approved shard"}
    _fake_s3(
        monkeypatch, {".metadata": b"modified", "__0_0.distcp": files["__0_0.distcp"]}
    )
    if approved_source:
        trust = _pin(tmp_path, files)
        _store(tmp_path, monkeypatch, source, trust)
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        load_distributed_checkpoint(
            torch.nn.Linear(1, 1, bias=False),
            source,
            credential_path="unused",
            local_cache_dir=str(tmp_path),
        )


def _fake_s3(
    monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]
) -> dict[str, Any]:
    import flashdreams.core.io.s3_filesystem as s3_module

    activity: dict[str, Any] = {
        "read_bytes": 0,
        "bodies_closed": set(),
        "client_closed": False,
    }

    class Body(io.BytesIO):
        def __init__(self, name: str) -> None:
            super().__init__(files[name])
            self.name = name

        def read(self, size: int = -1) -> bytes:
            data = super().read(size)
            activity["read_bytes"] += len(data)
            return data

        def close(self) -> None:
            activity["bodies_closed"].add(self.name)
            super().close()

    def get_object(*, Bucket: str, Key: str) -> dict[str, Body]:
        assert Bucket == "bucket"
        return {"Body": Body(Key.removeprefix("model/"))}

    class FakeS3FileSystem:
        def __init__(self, credential_path: str) -> None:
            assert credential_path == "unused"
            self.s3_client = SimpleNamespace(get_object=get_object)

        def _parse_s3_uri(self, path: str) -> tuple[str, str]:
            bucket, key = path.removeprefix("s3://").split("/", 1)
            return bucket, key

        def init_path(self, path: Any) -> str:
            return str(path)

        def concat_path(self, path: Any, suffix: str) -> str:
            return str(path).rstrip("/") + "/" + suffix

        def create_stream(self, path: Any, mode: str) -> io.BytesIO:
            assert mode == "rb"
            return io.BytesIO(files[str(path).removeprefix("s3://bucket/model/")])

        def close(self) -> None:
            activity["client_closed"] = True

    monkeypatch.setattr(s3_module, "S3FileSystem", FakeS3FileSystem)
    return activity


def test_s3_oversize_stream_is_bounded_and_closed_before_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flashdreams.core.io.s3_filesystem import S3StorageReader

    files = {".metadata": b"approved metadata", "__0_0.distcp": b"approved shard"}
    trust = _pin(tmp_path, files)
    activity = _fake_s3(monkeypatch, {**files, ".metadata": b"x" * 4096})
    _tripwire(monkeypatch)
    with pytest.raises(DCPTrustError):
        S3StorageReader("unused", "s3://bucket/model", dcp_trust=trust).read_metadata()
    assert activity["read_bytes"] == len(files[".metadata"]) + 1
    assert ".metadata" in activity["bodies_closed"]
    assert activity["client_closed"]


@pytest.mark.parametrize("tampered", [False, True])
def test_direct_s3_reader_enforces_trust_and_uses_snapshot(
    tampered: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flashdreams.core.io.s3_filesystem import S3StorageReader

    files = {".metadata": b"approved metadata", "__0_0.distcp": b"approved shard"}
    trust = _pin(tmp_path, files)
    source_files = dict(files)
    if tampered:
        source_files["__0_0.distcp"] = b"modified"
    _fake_s3(monkeypatch, source_files)
    if tampered:
        _tripwire(monkeypatch)
        with pytest.raises(DCPTrustError):
            S3StorageReader(
                "unused", "s3://bucket/model", dcp_trust=trust
            ).read_metadata()
    else:

        def inspect_snapshot(
            reader: FileSystemReader, *_args: Any, **_kwargs: Any
        ) -> bytes:
            return (Path(reader.path) / ".metadata").read_bytes()

        monkeypatch.setattr(FileSystemReader, "read_metadata", inspect_snapshot)
        with S3StorageReader("unused", "s3://bucket/model", dcp_trust=trust) as reader:
            source_files[".metadata"] = b"replacement"
            assert reader.read_metadata() == files[".metadata"]
