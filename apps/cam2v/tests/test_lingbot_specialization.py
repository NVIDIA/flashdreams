# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the integration-owned Lingbot Cam2V specialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli as tomllib

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_lingbot_registers_a_shared_cam2v_application() -> None:
    """Keep the entry point and dependency at the integration boundary."""
    manifest = tomllib.loads(
        (_REPO_ROOT / "integrations" / "lingbot" / "pyproject.toml").read_text()
    )

    assert "flashdreams-cam2v" in manifest["project"]["dependencies"]
    assert (
        manifest["project"]["entry-points"]["flashdreams.applications_v2"][
            "cam2v-lingbot"
        ]
        == "lingbot.cam2v.app:create_app"
    )
    assert (
        _REPO_ROOT / "integrations" / "lingbot" / "lingbot" / "cam2v" / "app.py"
    ).is_file()
