# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository checks preventing bundled HandDreams dataset assets."""

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu


def test_handdreams_dataset_directories_are_not_tracked() -> None:
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "internal/cosmos_interactive_internal/assets/agibot_scenes/",
            "internal/cosmos_interactive_internal/assets/agibot_scenario_gallery/",
            "apps/handdreams/handdreams/bundled_scenes/",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not tracked.stdout, f"Dataset assets must not be tracked:\n{tracked.stdout}"
