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

"""Local file and checkpoint helpers for the SANA-WM integration."""

from __future__ import annotations

import os
from pathlib import Path

from flashdreams.core.io.hf import maybe_download_hf_repo_on_rank0

HF_URI_SCHEME = "hf://"


def resolve_hf_path(path: str | Path) -> str:
    """Resolve a local path or ``hf://owner/repo/subpath`` URI to a local path.

    Remote repos are preloaded through
    :func:`~flashdreams.core.io.hf.maybe_download_hf_repo_on_rank0` so that
    multi-rank jobs do not race on the shared cache and so that the free-disk
    preflight applies, then resolved from that cache. Local paths and
    offline/local-only modes stay download-free.

    Args:
        path: Local path or ``hf://<owner>/<repo>[/<subpath>]`` URI.

    Returns:
        Local filesystem path to the artefact.

    Raises:
        ValueError: Malformed ``hf://`` URI.
    """
    path_str = str(path)
    if not path_str or Path(path_str).exists():
        return path_str
    if not path_str.startswith(HF_URI_SCHEME):
        return path_str

    from huggingface_hub import snapshot_download

    parts = path_str[len(HF_URI_SCHEME) :].split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid HF path {path_str!r}; expected hf://<owner>/<repo>[/<subpath>]."
        )
    repo_id = f"{parts[0]}/{parts[1]}"
    subpath = parts[2] if len(parts) > 2 else ""
    allow_patterns = None
    if subpath:
        allow_patterns = [subpath, f"{subpath}/*", f"{subpath}/**"]
    maybe_download_hf_repo_on_rank0(repo_id, allow_patterns=allow_patterns)
    local_root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_files_only=True,
    )
    return os.path.join(local_root, subpath) if subpath else local_root


__all__ = [
    "resolve_hf_path",
]
