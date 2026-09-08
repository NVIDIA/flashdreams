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

"""Default scene used by the driving applications."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_SCENE_UUID = "0d404ff7-2b66-498c-b047-1ed8cded60d4"
"""Scene UUID used by the driving applications."""

DEFAULT_SCENE_REPO_ID = "nvidia/omni-dreams-scenes"
"""Hugging Face dataset containing the default scene."""

DEFAULT_SCENE_FILENAME = f"scenes/clipgt-{DEFAULT_SCENE_UUID}.usdz"
"""Dataset-relative path of the default scene archive."""


def download_default_scene(
    download: Callable[..., Any] | None = None,
) -> Path:
    """Download the driving applications' built-in default USDZ scene.

    Args:
        download: Hugging Face download callable; ``None`` imports
            ``hf_hub_download`` lazily.

    Returns:
        Local Hugging Face cache path for the scene archive.

    Raises:
        RuntimeError: ``huggingface-hub`` is unavailable.
    """
    if download is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "The default driving scene requires huggingface-hub."
            ) from error
        download = hf_hub_download
    return Path(
        download(
            repo_id=DEFAULT_SCENE_REPO_ID,
            repo_type="dataset",
            filename=DEFAULT_SCENE_FILENAME,
        )
    )


__all__ = [
    "DEFAULT_SCENE_FILENAME",
    "DEFAULT_SCENE_REPO_ID",
    "DEFAULT_SCENE_UUID",
    "download_default_scene",
]
