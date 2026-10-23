# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-Auto transformer and LiDAR checkpoint preparation."""

from pathlib import Path
from urllib.parse import unquote, urlsplit

from cosmos_interactive import checkpoint_load
from huggingface_hub import hf_hub_download

from .config import CHECKPOINT_REPO_ID


def _download_lidar_checkpoint(source: str) -> Path | None:
    url = urlsplit(source)
    parts = url.path.strip("/").split("/")
    if (
        url.hostname in {"huggingface.co", "www.huggingface.co"}
        and len(parts) >= 5
        and "/".join(parts[:2]) == CHECKPOINT_REPO_ID
        and parts[2] in {"resolve", "blob"}
    ):
        revision = unquote(parts[3])
        hf_hub_download(
            repo_id=CHECKPOINT_REPO_ID,
            filename="lidar_vae/config.json",
            revision=revision,
        )
        return Path(
            hf_hub_download(
                repo_id=CHECKPOINT_REPO_ID,
                filename="lidar_vae/diffusion_pytorch_model.safetensors",
                revision=revision,
            )
        )
    return None


def resolve_lidar_checkpoint(
    source: str, *, checkpoint: str | Path | None = None, version: str = "v1p2"
) -> Path:
    """Select explicit codec weights or the matching Sim-Auto V1.2 companion.

    HF sources use the transformer's repository/revision and normal Hub cache.
    Local exports use their sibling ``lidar_vae`` directory. Explicit overrides
    retain their path and architecture; missing custom weights do not fall back.
    """
    if checkpoint is not None:
        return Path(checkpoint)
    if version != "v1p2":
        raise ValueError(
            "Automatic LiDAR weights are V1.2; v1p1 needs --lidar-checkpoint"
        )
    downloaded = _download_lidar_checkpoint(source)
    if downloaded is not None:
        return downloaded
    if not urlsplit(source).scheme:
        exported = Path(source).expanduser()
        directory = exported if exported.is_dir() else exported.parent
        companion = directory / "lidar_vae/diffusion_pytorch_model.safetensors"
        if companion.is_file():
            return companion
    raise ValueError(
        "No matching LiDAR companion for this model source; pass --lidar-checkpoint "
        "or use the Sim-Auto HF export with its lidar_vae directory"
    )


def load_exported_checkpoint(
    source: str, *, device: str
) -> checkpoint_load.CosmosMoTNetwork:
    """Cache matching HF LiDAR assets before loading the transformer."""
    _download_lidar_checkpoint(source)
    return checkpoint_load.load_exported_checkpoint(source, device=device)
