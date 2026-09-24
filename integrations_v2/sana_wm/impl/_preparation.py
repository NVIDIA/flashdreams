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

"""Preparation-only downloads for SANA-WM Hugging Face assets."""

from __future__ import annotations

import os

import huggingface_hub

from flashdreams.core.io.hf import maybe_download_hf_repo_on_rank0
from sana_wm.impl._tools import (
    HF_URI_SCHEME,
    _hf_uri_parts,
    _remember_prepared_hf_path,
)
from sana_wm.impl.constants import (
    SANA_WM_STREAMING_CAUSAL_VAE_ROOT,
    SANA_WM_STREAMING_REFINER_GEMMA_ROOT,
    SANA_WM_STREAMING_REFINER_ROOT,
)


def preload_hf_path(path: str) -> str:
    """Download an ``hf://`` path and record its local runtime path.

    Args:
        path: Local path or ``hf://<owner>/<repo>[/<subpath>]`` URI.

    Returns:
        Local filesystem path to the prepared artifact.

    Raises:
        ValueError: Malformed ``hf://`` URI.
    """
    if not path.startswith(HF_URI_SCHEME):
        return path

    repo_id, subpath, allow_patterns = _hf_uri_parts(path)
    maybe_download_hf_repo_on_rank0(repo_id, allow_patterns=allow_patterns)
    local_root = huggingface_hub.snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_files_only=True,
    )
    local_path = os.path.join(local_root, subpath) if subpath else local_root
    _remember_prepared_hf_path(path, local_path)
    return local_path


def preload_sana_wm_streaming_paths() -> None:
    """Prepare every Hugging Face path used lazily by SANA-WM streaming."""
    for path in (
        SANA_WM_STREAMING_CAUSAL_VAE_ROOT,
        SANA_WM_STREAMING_REFINER_ROOT,
        SANA_WM_STREAMING_REFINER_GEMMA_ROOT,
    ):
        preload_hf_path(path)


__all__ = ["preload_sana_wm_streaming_paths"]
