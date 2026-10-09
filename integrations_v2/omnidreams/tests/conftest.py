# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SCENE_ZIP = REPO_ROOT / "assets" / "example_data" / "omnidreams" / "clipgt.zip"


@pytest.fixture(scope="session")
def example_scene_zip_path() -> Path:
    if not EXAMPLE_SCENE_ZIP.exists():
        pytest.skip(
            f"Missing integration-test scene archive at {EXAMPLE_SCENE_ZIP}. "
            "Run assets/download.sh to fetch test data."
        )
    return EXAMPLE_SCENE_ZIP


@pytest.fixture(scope="session")
def example_scene_zip_bytes(example_scene_zip_path: Path) -> bytes:
    return example_scene_zip_path.read_bytes()
