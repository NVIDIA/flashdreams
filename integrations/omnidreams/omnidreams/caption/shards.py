# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebDataset-compatible tar shard writer for the caption capture harness.

Each captured window is one WebDataset *sample*: a set of files sharing a
dot-free key, e.g. ``000123.latents.npy`` / ``000123.rgb.jpg`` / ``000123.json``.
Samples are packed into rolling tar shards (``raw-000000.tar`` …) and a
top-level ``manifest.jsonl`` records one line per sample for quick indexing and
label joins downstream.

Deliberately stdlib-only (``tarfile``): the capture host writes shards without a
hard WebDataset dependency, and any later loader (WebDataset, or a plain tar
reader) can consume them. Callers hand in already-encoded bytes per field, so
this module stays agnostic to how latents / RGB are serialized.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any


class ShardWriter:
    """Write capture samples into rolling, WebDataset-compatible tar shards.

    Args:
        out_dir: Directory to hold the ``*.tar`` shards and ``manifest.jsonl``.
        prefix: Shard filename prefix (``{prefix}-{index:06d}.tar``).
        max_per_shard: Samples per shard before rolling to the next file.
        start_shard: First shard index (for resuming a capture).
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        prefix: str = "raw",
        max_per_shard: int = 1000,
        start_shard: int = 0,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._prefix = prefix
        self._max_per_shard = max_per_shard
        self._shard_index = start_shard
        self._in_shard = 0
        self._total = 0
        self._tar: tarfile.TarFile | None = None
        self._manifest = (self._out_dir / "manifest.jsonl").open("a", encoding="utf-8")

    def _shard_name(self) -> str:
        return f"{self._prefix}-{self._shard_index:06d}.tar"

    def _roll(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._shard_index += 1
        self._tar = tarfile.open(self._out_dir / self._shard_name(), "w")
        self._in_shard = 0

    def write(
        self, key: str, parts: dict[str, bytes], meta: dict[str, Any] | None = None
    ) -> str:
        """Add one sample (its per-field byte blobs) and a manifest line.

        Args:
            key: Dot-free sample key (WebDataset groups members by the substring
                before the first ``.``; a dot in the key would split the sample).
            parts: Field extension -> encoded bytes, e.g.
                ``{"latents.npy": ..., "rgb.jpg": ..., "json": ...}``.
            meta: Extra fields merged into the manifest line for this sample.

        Returns:
            The shard filename the sample was written into.
        """
        if "." in key:
            raise ValueError(f"sample key must be dot-free, got {key!r}")
        if self._tar is None or self._in_shard >= self._max_per_shard:
            self._roll()
        assert self._tar is not None

        for ext, data in parts.items():
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            info.mtime = 0  # deterministic shards (no wall-clock in the tar)
            self._tar.addfile(info, io.BytesIO(data))

        shard = self._shard_name()
        line = {"__key__": key, "shard": shard, "fields": sorted(parts)}
        if meta:
            line.update(meta)
        self._manifest.write(json.dumps(line) + "\n")
        self._manifest.flush()
        self._in_shard += 1
        self._total += 1
        return shard

    @property
    def total(self) -> int:
        """Number of samples written so far."""
        return self._total

    def close(self) -> None:
        """Finalize the open shard and the manifest."""
        if self._tar is not None:
            self._tar.close()
            self._tar = None
        self._manifest.close()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
