# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face access helpers for evaluation data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omnidreams.eval.manifest import AssetRef


def list_hf_dataset_files(
    *,
    repo_id: str,
    revision: str,
    subpath: str,
    token: str | None = None,
) -> list[Any]:
    """List dataset files below ``subpath`` without downloading file blobs."""

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError("omnidreams-eval discover requires huggingface_hub") from exc

    api = HfApi(token=token)
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=subpath.strip("/"),
        recursive=True,
    )
    return [entry for entry in entries if _entry_is_file(entry)]


def download_asset(
    asset: AssetRef,
    *,
    repo_id: str,
    revision: str,
    destination_root: Path,
    token: str | None = None,
    force: bool = False,
) -> Path:
    """Download one remote asset into ``destination_root`` and return its path."""

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "omnidreams-eval stage-batch requires huggingface_hub"
        ) from exc

    local_path = destination_root / asset.path
    if (
        not force
        and local_path.exists()
        and (asset.size is None or local_path.stat().st_size == asset.size)
    ):
        return local_path
    destination_root.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=asset.path,
            token=token,
            local_dir=destination_root,
            force_download=force,
        )
    )


def _entry_is_file(entry: Any) -> bool:
    entry_type = getattr(entry, "type", None)
    if entry_type is not None:
        return entry_type == "file"
    return bool(getattr(entry, "path", None) or getattr(entry, "rfilename", None))
