# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3: assemble capture shards + control labels into train/val tensors.

Reads the WebDataset tar shards written by ``capture.py`` and their
``manifest.jsonl``, splits **by session** (not by window — sessions are whole
rollouts, so a window-level split would leak), computes inverse-frequency class
weights, and pre-loads each window's latents into RAM for fast epochs.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from omnidreams.caption.model import CAPTION_CLASSES  # noqa: F401 (re-exported)


def class_index(classes: list[str]) -> dict[str, int]:
    """Map each class name to its logits index."""
    return {c: i for i, c in enumerate(classes)}


def read_manifest(shards_dir: str | Path, classes: list[str]) -> list[dict]:
    """One dict (key/shard/session/label) per window whose label is in ``classes``."""
    keep = set(classes)
    items: list[dict] = []
    path = Path(shards_dir) / "manifest.jsonl"
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            label = d.get("label", d.get("class_hint"))  # new schema; old fallback
            if label in keep:
                items.append(
                    {
                        "key": d["__key__"],
                        "shard": d["shard"],
                        "session": d.get("session_id", "unknown"),
                        "label": label,
                    }
                )
    return items


def sessions_of(items: list[dict]) -> list[str]:
    """Sorted unique session ids present in ``items``."""
    return sorted({it["session"] for it in items})


def session_split(
    items: list[dict], val_sessions: list[str]
) -> tuple[list[dict], list[dict]]:
    """Split into (train, val) by holding out whole ``val_sessions``."""
    val = set(val_sessions)
    train = [it for it in items if it["session"] not in val]
    held = [it for it in items if it["session"] in val]
    return train, held


def class_weights(
    items: list[dict],
    classes: list[str],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Inverse-frequency class weights (mean 1) for a weighted cross-entropy."""
    c2i = class_index(classes)
    counts = np.zeros(len(classes), dtype=np.float64)
    for it in items:
        counts[c2i[it["label"]]] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    tensor = torch.tensor(weights, dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor


class InMemoryLatentDataset(Dataset):
    """Pre-loads each window's ``latents.npy`` into one RAM tensor (fp16).

    ``__getitem__`` returns ``(latent[T,C,H,W] fp32, label)``. The one-time load
    reads the shards once; epochs then index RAM, so training is not I/O-bound.
    """

    def __init__(
        self, shards_dir: str | Path, items: list[dict], classes: list[str]
    ) -> None:
        shards_dir = Path(shards_dir)
        c2i = class_index(classes)
        self.labels = torch.tensor(
            [c2i[it["label"]] for it in items], dtype=torch.long
        )
        self.latents: torch.Tensor | None = None
        tars: dict[str, tarfile.TarFile] = {}
        try:
            for i, it in enumerate(items):
                shard = it["shard"]
                if shard not in tars:
                    tars[shard] = tarfile.open(shards_dir / shard)
                blob = tars[shard].extractfile(f"{it['key']}.latents.npy").read()
                arr = np.load(io.BytesIO(blob))  # [T, C, H, W] fp16
                if self.latents is None:
                    self.latents = torch.empty(
                        (len(items), *arr.shape), dtype=torch.float16
                    )
                self.latents[i] = torch.from_numpy(arr)
        finally:
            for tar in tars.values():
                tar.close()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.latents is not None
        return self.latents[i].float(), self.labels[i]
