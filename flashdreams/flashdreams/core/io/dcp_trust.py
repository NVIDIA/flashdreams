# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned DCP manifests and private verified checkpoint snapshots.

Approve a checkpoint where it was produced with
``python -m flashdreams.core.io.dcp_trust pin <dcp-dir>``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import weakref
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Iterator

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.filesystem import FileSystem, FileSystemBase

from flashdreams.core.io.disk import (
    DiskSpaceContext,
    ensure_free_disk,
    raise_if_disk_space_error,
)

_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
"""Bound JSON trust records to 4 MiB; checkpoint tensors are streamed separately."""

_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
"""Open without waiting for a FIFO writer; Windows has no filesystem FIFOs."""

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_DIR_FD = os.open in os.supports_dir_fd and hasattr(os, "O_DIRECTORY")
"""Anchor snapshots on a directory descriptor; Windows has no ``dir_fd``."""

_PIN_COMMAND = "python -m flashdreams.core.io.dcp_trust pin <dcp-dir>"


class DCPTrustError(PermissionError):
    """An unapproved or modified DCP cannot enter the metadata parser."""


@dataclass(frozen=True)
class DCPTrust:
    """Identify a manifest with an independently approved SHA-256 pin."""

    manifest_path: str
    """Local JSON manifest path, supplied separately from the checkpoint source."""

    sha256: str
    """Expected manifest digest from a trusted publisher or controlled export."""


def _source_id(path: str | os.PathLike) -> str:
    text = str(path)
    if text.startswith("s3://"):
        return text.rstrip("/")
    if "://" in text:
        raise DCPTrustError("Only local and S3 DCP sources are supported")
    # Do not resolve symlinks: a source replacement must not change the trust key.
    return os.path.abspath(os.path.expanduser(text))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DCPTrustError("Duplicate key in DCP trust record")
        result[key] = value
    return result


def _open_regular(path: str | os.PathLike) -> io.BufferedReader:
    """Open replaceable local storage; a FIFO or device fails instead of blocking."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | _NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DCPTrustError(f"DCP trust input {path} is not a regular file")
        if _NONBLOCK:
            os.set_blocking(fd, True)
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _read_record(path: str, label: str, *, private: bool = False) -> bytes:
    try:
        if private and os.name != "nt":
            # Without the sticky bit, anyone who can write a directory can swap
            # its entries, so every directory above the real store is checked
            # and the store is opened by that checked path.
            path = os.path.realpath(path)
            for parent in Path(path).parents:
                mode = os.stat(parent).st_mode
                if mode & (stat.S_IWGRP | stat.S_IWOTH) and not mode & stat.S_ISVTX:
                    raise DCPTrustError(
                        f"{label} directory {parent} must not be group- or "
                        f"world-writable; run: chmod go-w {parent}"
                    )
        with _open_regular(path) as stream:
            mode = os.fstat(stream.fileno()).st_mode
            # Windows reports synthetic group/other bits; ACLs govern access there.
            if private and os.name != "nt" and mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise DCPTrustError(
                    f"{label} {path} must not be group- or world-writable; "
                    f"run: chmod 600 {path}"
                )
            data = stream.read(_MAX_MANIFEST_BYTES + 1)
    except DCPTrustError:
        raise
    except OSError as exc:
        raise DCPTrustError(f"Cannot read {label} {path}") from exc
    if len(data) > _MAX_MANIFEST_BYTES:
        raise DCPTrustError(f"{label} {path} exceeds 4 MiB")
    return data


def _json_record(data: bytes, label: str) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise DCPTrustError(f"Invalid JSON in {label}") from exc


def default_trust_store() -> Path:
    """Return the per-user store used when ``FLASHDREAMS_DCP_TRUST_STORE`` is unset."""
    config = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(os.path.expanduser(config)) / "flashdreams" / "dcp-trust.json"


def _store_path() -> str:
    configured = os.environ.get("FLASHDREAMS_DCP_TRUST_STORE")
    return os.path.expanduser(configured or str(default_trust_store()))


def _load_store(path: str) -> dict[str, Any]:
    label = f"DCP trust store {path}"
    store = _json_record(_read_record(path, "DCP trust store", private=True), label)
    if not isinstance(store, dict):
        raise DCPTrustError(f"{label} must map source IDs to manifest pins")
    return store


def _trust_store(source: str) -> dict[str, Any]:
    path = _store_path()
    if not os.environ.get("FLASHDREAMS_DCP_TRUST_STORE") and not os.path.lexists(path):
        raise DCPTrustError(
            f"DCP {source} requires an independently pinned manifest. Where the "
            f"checkpoint was produced, run: {_PIN_COMMAND}; see the checkpoint "
            "trust guide."
        )
    return _load_store(path)


def _trust_from_store(source: str, store: dict[str, Any]) -> DCPTrust:
    if source not in store:
        raise DCPTrustError(
            f"DCP source {source} is not listed in the DCP trust store; where the "
            f"checkpoint was produced, run: {_PIN_COMMAND}"
        )
    record = store[source]
    if (
        not isinstance(record, dict)
        or set(record) != {"manifest_path", "sha256"}
        or not isinstance(record["manifest_path"], str)
        or not os.path.isabs(record["manifest_path"])
    ):
        raise DCPTrustError(
            f"Invalid DCP trust store record for {source}: expected only an "
            "absolute 'manifest_path' and a 'sha256'"
        )
    return DCPTrust(**record)


def resolve_dcp_checkpoint_path(checkpoint_path: str) -> str:
    """Select an approved DCP root or its approved ``model`` subdirectory.

    Consult only the operator's trust store, without parsing checkpoint bytes.
    A selected root's verification failure never falls back to another source.
    """
    base = _source_id(checkpoint_path)
    store = _trust_store(base)
    selected = base if base in store else base + "/model"
    if selected not in store:
        raise DCPTrustError(
            f"Neither DCP source {base} nor {selected} is listed in the DCP "
            f"trust store; where the checkpoint was produced, run: {_PIN_COMMAND}"
        )
    _trust_from_store(selected, store)
    return selected


def _manifest_files(path: str, trust: DCPTrust | None) -> dict[str, Any]:
    if trust is None:
        trust = _trust_from_store(_source_id(path), _trust_store(path))
    if not isinstance(trust, DCPTrust) or not _valid_digest(trust.sha256):
        raise DCPTrustError(f"DCP {path} manifest requires a SHA-256 pin")
    label = f"pinned DCP manifest {trust.manifest_path}"
    data = _read_record(trust.manifest_path, "pinned DCP manifest")
    actual = hashlib.sha256(data).hexdigest()
    if actual != trust.sha256.lower():
        raise DCPTrustError(
            f"DCP manifest SHA-256 mismatch for {trust.manifest_path}: "
            f"pinned {trust.sha256.lower()}, got {actual}"
        )
    manifest = _json_record(data, label)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "files"}
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or not isinstance(manifest["files"], dict)
        or not manifest["files"]
    ):
        raise DCPTrustError(f"Invalid DCP manifest schema in {label}")
    files = manifest["files"]
    for name, entry in files.items():
        if (
            not name
            or any(part in ("", ".", "..") for part in name.split("/"))
            or "\\" in name
            or ":" in name
            or "\x00" in name
            or not isinstance(entry, dict)
            or set(entry) != {"sha256", "size"}
            or not _valid_digest(entry["sha256"])
            or type(entry["size"]) is not int
            or not 0 <= entry["size"] <= 2**63 - 1
        ):
            raise DCPTrustError(f"Invalid DCP manifest file entry {name!r} in {label}")
    if not any(re.fullmatch(r"(?:__[0-9]+)?\.metadata", name) for name in files):
        raise DCPTrustError(f"{label} must pin metadata")
    return files


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
    )


def _open_snapshot_root(root: Path) -> int | None:
    """Hold the new snapshot directory so a rename of its path cannot swap it."""
    if not _DIR_FD:
        return None
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        os.close(fd)
        raise DCPTrustError(f"DCP snapshot directory {root} was replaced before use")
    return fd


def _remove_snapshot(root: Path, root_fd: int | None) -> None:
    """Delete the snapshot through the held directory, even if it was renamed."""
    if root_fd is None:
        # Windows refuses to delete the read-only snapshot files; clear the flag.
        for directory, _, names in os.walk(root):
            for name in names:
                with suppress(OSError):
                    os.chmod(
                        os.path.join(directory, name), stat.S_IWRITE | stat.S_IREAD
                    )
        shutil.rmtree(root, ignore_errors=True)
        return
    with suppress(OSError):
        for _, dirs, names, fd in os.fwalk(topdown=False, dir_fd=root_fd):
            for name in names:
                os.unlink(name, dir_fd=fd)
            for name in dirs:
                os.rmdir(name, dir_fd=fd)
        # The emptied directory goes only if its path still names it.
        if os.path.samestat(os.stat(root, follow_symlinks=False), os.fstat(root_fd)):
            os.rmdir(root)
    os.close(root_fd)


def _snapshot_name(root: Path, root_fd: int | None, name: str) -> str:
    return name if root_fd is not None else str(root / name)


def _create_snapshot_file(root: Path, root_fd: int | None, name: str) -> io.IOBase:
    """Create one owner-read-only snapshot file inside the held directory."""
    parts = name.split("/")
    for depth in range(1, len(parts)):
        with suppress(FileExistsError):
            os.mkdir(
                _snapshot_name(root, root_fd, "/".join(parts[:depth])),
                0o700,
                dir_fd=root_fd,
            )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
    flags |= getattr(os, "O_BINARY", 0)
    fd = os.open(_snapshot_name(root, root_fd, name), flags, 0o400, dir_fd=root_fd)
    return os.fdopen(fd, "wb")


class _SnapshotFileSystem(FileSystem):
    def __init__(self, root: Path, root_fd: int | None, files: dict[str, Any]) -> None:
        self.root = root
        self.root_fd = root_fd
        self.files = frozenset(files)

    def check_path(self, path: str | os.PathLike) -> str:
        try:
            name = Path(path).relative_to(self.root).as_posix()
        except ValueError as exc:
            raise DCPTrustError("DCP read escaped the verified snapshot") from exc
        if name not in self.files:
            raise DCPTrustError(
                f"DCP read {name!r} is not covered by the pinned manifest"
            )
        return name

    @contextmanager
    def create_stream(self, path: str | os.PathLike, mode: str) -> Iterator[io.IOBase]:
        name = self.check_path(path)
        if mode != "rb":
            raise DCPTrustError("Verified DCP snapshots are read-only")
        flags = os.O_RDONLY | _NOFOLLOW | getattr(os, "O_BINARY", 0)
        # Open through the held directory, not the path: renaming the
        # snapshot (e.g. in a shared, non-sticky TMPDIR) cannot swap bytes.
        fd = os.open(
            _snapshot_name(self.root, self.root_fd, name), flags, dir_fd=self.root_fd
        )
        with os.fdopen(fd, "rb") as stream:
            yield stream

    def _unsupported(self, *_args: Any, **_kwargs: Any) -> Any:
        raise DCPTrustError("Verified DCP snapshots allow only pinned reads")

    exists = ls = rename = mkdir = rm_file = _unsupported

    def close(self) -> None:
        """Stop reads; the reader deletes the snapshot and releases the directory."""
        self.root_fd = -1


class VerifiedDCPReader(FileSystemReader):
    """Read DCP only from a private copy verified before metadata parsing.

    All listed metadata and shards are verified eagerly. Source/cache changes
    after construction cannot change the bytes consumed by PyTorch. Use as a
    context manager or call ``close`` after the final tensor read.
    """

    fs: _SnapshotFileSystem
    """Allow reads only from files covered by the pinned manifest."""

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        trust: DCPTrust | None = None,
        source_fs: FileSystemBase | None = None,
    ) -> None:
        self._source = _source_id(path)
        files = _manifest_files(self._source, trust)
        source_fs = source_fs if source_fs is not None else FileSystem()
        disk: DiskSpaceContext = dict(
            label="verified DCP snapshot",
            required_bytes=sum(entry["size"] for entry in files.values()),
            env_vars=("TMPDIR",),
            settings={"DCP source": self._source},
        )
        ensure_free_disk(tempfile.gettempdir(), **disk)
        root = Path(tempfile.mkdtemp(prefix="flashdreams-dcp-"))
        root_fd = None
        try:
            root_fd = _open_snapshot_root(root)
            for name, entry in files.items():
                digest = hashlib.sha256()
                remaining = entry["size"]
                source = source_fs.concat_path(self._source, name)
                with (
                    self._open_source(source_fs, source) as src,
                    _create_snapshot_file(root, root_fd, name) as dst,
                ):
                    while chunk := src.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(chunk)
                        if remaining < 0:
                            raise DCPTrustError(
                                f"DCP file {name!r} in {self._source} exceeds its "
                                f"pinned size of {entry['size']} bytes"
                            )
                        digest.update(chunk)
                        dst.write(chunk)
                if remaining or digest.hexdigest() != entry["sha256"].lower():
                    raise DCPTrustError(
                        f"DCP file {name!r} in {self._source} does not match its pin: "
                        f"expected {entry['size']} bytes, SHA-256 "
                        f"{entry['sha256'].lower()}; got {entry['size'] - remaining} "
                        f"bytes, SHA-256 {digest.hexdigest()}"
                    )
            super().__init__(root)
            self.fs = _SnapshotFileSystem(root, root_fd, files)
            # Runs once, from close() or when an unclosed reader is collected.
            self._remove = weakref.finalize(self, _remove_snapshot, root, root_fd)
        except BaseException as exc:
            _remove_snapshot(root, root_fd)
            if isinstance(exc, OSError) and not isinstance(exc, DCPTrustError):
                # Snapshot writes (scratch exhaustion) are not trust failures.
                raise_if_disk_space_error(exc, path=root, **disk)
                raise DCPTrustError(
                    f"Cannot read all pinned DCP files from {self._source}"
                ) from exc
            raise

    def _open_source(
        self, source_fs: FileSystemBase, path: str | os.PathLike
    ) -> ContextManager[io.IOBase]:
        if isinstance(source_fs, FileSystem):
            return _open_regular(path)
        return source_fs.create_stream(path, "rb")

    def read_metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Parse only metadata covered by the verified snapshot."""
        rank = kwargs.get("rank")
        filename = ".metadata" if rank is None else f"__{rank}.metadata"
        self.fs.check_path(self.fs.concat_path(self.path, filename))
        metadata = super().read_metadata(*args, **kwargs)
        storage_data = getattr(metadata, "storage_data", None)
        if storage_data is not None:
            for entry in storage_data.values():
                self.fs.check_path(self.fs.concat_path(self.path, entry.relative_path))
        return metadata

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        """Reset planning state without redirecting to an unverified source."""
        if checkpoint_id is not None and _source_id(checkpoint_id) not in (
            self._source,
            str(self.path),
        ):
            raise DCPTrustError("Cannot redirect a verified DCP reader")
        super().reset()

    def close(self) -> None:
        """Delete the private checkpoint snapshot."""
        self.fs.close()
        self._remove()

    def __enter__(self) -> VerifiedDCPReader:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _write_private(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with an owner-only file in an owner-only folder."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(staging, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(staging)
        raise


def _manifest_for(root: Path) -> bytes:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DCPTrustError(f"Cannot pin symlink {path}; pin regular files only")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        size = 0
        with _open_regular(path) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        files[path.relative_to(root).as_posix()] = {
            "sha256": digest.hexdigest(),
            "size": size,
        }
    return json.dumps({"version": 1, "files": files}, sort_keys=True).encode()


def main(argv: list[str] | None = None) -> int:
    """Pin a DCP manifest in the trust store (``pin`` command)."""
    parser = argparse.ArgumentParser(prog="python -m flashdreams.core.io.dcp_trust")
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser(
        "pin",
        help="approve one DCP checkpoint",
        description=(
            "Approve a DCP checkpoint by pinning its manifest. Run this where the "
            "checkpoint was produced, or with a publisher's manifest and digest "
            "received through a trusted channel. Approval trusts the checkpoint "
            "metadata as code: never pin an unknown download just to silence a "
            "verification error."
        ),
    )
    pin.add_argument(
        "path",
        help="DCP component directory to hash; with --manifest, the source ID",
    )
    pin.add_argument(
        "--source", help="approve this source ID instead, e.g. its s3:// destination"
    )
    pin.add_argument("--manifest", help="register this publisher manifest")
    pin.add_argument("--sha256", help="its independently received SHA-256")
    pin.add_argument(
        "--store",
        help="trust store to update (default: FLASHDREAMS_DCP_TRUST_STORE, "
        f"else {default_trust_store()})",
    )
    args = parser.parse_args(argv)
    if (args.manifest is None) != (args.sha256 is None):
        parser.error("--manifest and --sha256 must be given together")
    if args.manifest is not None and args.source is not None:
        parser.error("with --manifest, PATH is the source ID")
    store = Path(os.path.expanduser(args.store or _store_path())).absolute()
    try:
        if args.manifest is not None:
            source = _source_id(args.path)
            data = _read_record(args.manifest, "DCP manifest")
            if hashlib.sha256(data).hexdigest() != args.sha256.lower():
                raise DCPTrustError(
                    f"{args.manifest} does not match --sha256; obtain both "
                    "through a trusted channel"
                )
        else:
            root = Path(args.path)
            if not root.is_dir():
                raise DCPTrustError(f"{root} is not a DCP directory")
            source = _source_id(args.source or root)
            data = _manifest_for(root)
        digest = hashlib.sha256(data).hexdigest()
        trust = DCPTrust(str(store.parent / "dcp-pins" / f"{digest}.json"), digest)
        store.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_private(Path(trust.manifest_path), data)
        _manifest_files(source, trust)  # same schema checks as loading
        # ponytail: read-modify-write; concurrent pins to one store keep the last.
        entries = _load_store(str(store)) if os.path.lexists(store) else {}
        entries[source] = {"manifest_path": trust.manifest_path, "sha256": digest}
        _write_private(store, json.dumps(entries, indent=2, sort_keys=True).encode())
        _trust_from_store(source, _load_store(str(store)))  # loader's own checks
    except (DCPTrustError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Pinned {source}\n  manifest {trust.manifest_path}\n  sha256   {digest}")
    if str(store) != _store_path():
        print(f"  Loaders use this store with FLASHDREAMS_DCP_TRUST_STORE={store}")
    if args.manifest is None:
        print(
            "  On another machine, copy the manifest, send the SHA-256 separately, "
            f"and run: python -m flashdreams.core.io.dcp_trust pin --manifest FILE "
            f"--sha256 {digest} {source}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
