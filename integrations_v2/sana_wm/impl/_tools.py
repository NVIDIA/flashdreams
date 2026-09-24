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

from pathlib import Path

HF_URI_SCHEME = "hf://"

_PREPARED_HF_PATHS: dict[str, str] = {}
"""Process-local paths populated during SANA-WM application initialization."""


def _hf_uri_parts(path: str) -> tuple[str, str, list[str] | None]:
    """Parse an ``hf://`` URI into its repository and optional subpath."""
    parts = path[len(HF_URI_SCHEME) :].split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid HF path {path!r}; expected hf://<owner>/<repo>[/<subpath>]."
        )
    repo_id = f"{parts[0]}/{parts[1]}"
    subpath = parts[2] if len(parts) > 2 else ""
    allow_patterns = None
    if subpath:
        allow_patterns = [subpath, f"{subpath}/*", f"{subpath}/**"]
    return repo_id, subpath, allow_patterns


def _remember_prepared_hf_path(path: str, local_path: str) -> None:
    _PREPARED_HF_PATHS[path] = local_path


def resolve_hf_path(path: str | Path) -> str:
    """Resolve a local path or ``hf://owner/repo/subpath`` URI to a local path.

    Remote paths must be prepared during application initialization. Runtime
    callers only read the process-local prepared-path cache.

    Args:
        path: Local path or ``hf://<owner>/<repo>[/<subpath>]`` URI.

    Returns:
        Local filesystem path to the artefact.

    Raises:
        ValueError: Malformed ``hf://`` URI.
        RuntimeError: The remote path was not prepared during initialization.
    """
    path_str = str(path)
    if not path_str or Path(path_str).exists():
        return path_str
    if not path_str.startswith(HF_URI_SCHEME):
        return path_str

    _hf_uri_parts(path_str)
    try:
        return _PREPARED_HF_PATHS[path_str]
    except KeyError as exc:
        raise RuntimeError(
            f"SANA-WM Hugging Face path {path_str!r} was not prepared during "
            "application initialization."
        ) from exc


__all__ = [
    "resolve_hf_path",
]
