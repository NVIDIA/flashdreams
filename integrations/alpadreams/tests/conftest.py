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

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SCENE_ZIP = REPO_ROOT / "assets" / "example_data" / "alpadreams" / "clipgt.zip"


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
