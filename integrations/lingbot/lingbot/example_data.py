# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled LingBot-World example-data helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache

EXAMPLE_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/Robbyant/lingbot-world-v2/main/examples"
)
"""HTTP base URL for the canonical examples shared by all LingBot versions."""

EXAMPLE_DATA_DIR_LOCAL = default_flashdreams_cache_dir() / "example_data/lingbot_world"
"""Local cache root where downloaded example folders are stored."""

EXAMPLE_DATA_FILENAMES = (
    "image.jpg",
    "poses.npy",
    "intrinsics.npy",
    "prompt.txt",
)
"""Example assets downloaded when each file is available upstream."""

EXAMPLE_DATA_AVAILABLE_IDXS = (0, 1, 2, 3, 4, 5)
"""Supported upstream example indices currently hosted under ``examples/``."""

EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS = (0, 1, 2, 5)
"""Example indices that provide their own upstream ``prompt.txt`` file."""


def example_data_dirname(example_idx: int) -> str:
    """Format ``example_idx`` into the upstream folder naming convention."""
    assert example_idx in EXAMPLE_DATA_AVAILABLE_IDXS, (
        f"--example_idx must be one of {EXAMPLE_DATA_AVAILABLE_IDXS}."
    )
    return f"{example_idx:02d}"


def example_asset_urls(example_idx: int) -> dict[str, str]:
    """Return canonical upstream URLs for a Lingbot example."""
    dirname = example_data_dirname(example_idx)
    return {
        "image": f"{EXAMPLE_DATA_BASE_URL}/{dirname}/image.jpg",
        "intrinsics": f"{EXAMPLE_DATA_BASE_URL}/{dirname}/intrinsics.npy",
        "poses": f"{EXAMPLE_DATA_BASE_URL}/{dirname}/poses.npy",
    }


def ensure_example_data_downloaded(*, is_rank_zero: bool, example_idx: int) -> Path:
    """Download bundled GitHub example files on rank 0; barrier other ranks."""
    example_dirname = example_data_dirname(example_idx)
    cache_dir = EXAMPLE_DATA_DIR_LOCAL / example_dirname
    if is_rank_zero:
        for filename in EXAMPLE_DATA_FILENAMES:
            if (
                filename == "prompt.txt"
                and example_idx not in EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS
            ):
                continue
            download_to_cache(
                f"{EXAMPLE_DATA_BASE_URL}/{example_dirname}/{filename}",
                cache_dir=cache_dir,
                filename=filename,
            )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return cache_dir


__all__ = [
    "EXAMPLE_DATA_AVAILABLE_IDXS",
    "EXAMPLE_DATA_BASE_URL",
    "EXAMPLE_DATA_DIR_LOCAL",
    "EXAMPLE_DATA_FILENAMES",
    "EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS",
    "ensure_example_data_downloaded",
    "example_asset_urls",
    "example_data_dirname",
]
