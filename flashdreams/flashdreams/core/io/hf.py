# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hugging Face helpers shared across encoders.

Remote repos are preloaded before ``from_pretrained(..., local_files_only=True)``
so multi-rank jobs do not race to download the same snapshot or treat a partial
cache entry as complete.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import torch.distributed as dist
from filelock import FileLock
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from flashdreams.core.distributed import get_global_rank, is_distributed_initialized


def _str2bool(v: str | bool) -> bool:
    """Parse the usual yes/no/true/false/1/0 strings into a bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError(f"Boolean value expected, got {v!r}")


def _hub_cache_dir(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    return Path(HUGGINGFACE_HUB_CACHE).expanduser()


def _lock_path(
    repo_id: str,
    revision: str | None,
    cache_dir: str | os.PathLike[str] | None,
) -> Path:
    cache_root = _hub_cache_dir(cache_dir)
    lock_key = f"{repo_id}@{revision or 'main'}"
    lock_digest = hashlib.sha256(lock_key.encode("utf-8")).hexdigest()[:16]
    safe_name = repo_id.replace("/", "--")
    locks_dir = cache_root / ".flashdreams_locks"
    return locks_dir / f"{safe_name}-{lock_digest}.lock"


def _normalize_patterns(
    patterns: str | Sequence[str] | None,
) -> str | list[str] | None:
    if patterns is None or isinstance(patterns, str):
        return patterns
    return list(patterns)


def _download_snapshot(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | os.PathLike[str] | None,
    allow_patterns: str | Sequence[str] | None,
    ignore_patterns: str | Sequence[str] | None,
) -> None:
    lock_file = _lock_path(repo_id, revision, cache_dir)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_file)):
        snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=False,
            allow_patterns=_normalize_patterns(allow_patterns),
            ignore_patterns=_normalize_patterns(ignore_patterns),
        )


def maybe_download_hf_repo_on_rank0(
    repo_id_or_path: str,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    allow_patterns: str | Sequence[str] | None = None,
    ignore_patterns: str | Sequence[str] | None = None,
) -> None:
    """Download a remote HF repo snapshot from rank 0 when downloads are allowed.

    Local paths and explicit offline/local-only modes are no-ops. For remote
    repositories, rank 0 preloads the snapshot while other distributed ranks
    wait for its success/failure signal. A filesystem lock serializes
    independent processes that share the same HF cache directory.
    """
    if (
        os.path.isdir(repo_id_or_path)
        or _str2bool(os.getenv("HF_HUB_OFFLINE", "false"))
        or _str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))
    ):
        return

    rank = get_global_rank()
    payload: list[dict[str, str | None]]
    if rank == 0:
        try:
            _download_snapshot(
                repo_id_or_path,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            payload = [{"error": None}]
        except Exception as exc:
            payload = [{"error": f"{type(exc).__name__}: {exc}"}]
    else:
        payload = [{"error": None}]

    if is_distributed_initialized():
        dist.broadcast_object_list(payload, src=0)

    error = payload[0]["error"]
    if error is not None:
        raise RuntimeError(
            f"Rank 0 failed to download Hugging Face repo {repo_id_or_path!r}: {error}"
        )
