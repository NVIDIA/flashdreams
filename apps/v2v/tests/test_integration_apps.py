# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import tomli as tomllib

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_flashvsr_owns_its_v2v_application() -> None:
    project_dir = _REPO_ROOT / "integrations" / "flashvsr"
    manifest = tomllib.loads((project_dir / "pyproject.toml").read_text())

    assert (project_dir / "flashvsr" / "v2v" / "app.py").is_file()
    assert (
        manifest["project"]["entry-points"]["flashdreams.applications"]["v2v-flashvsr"]
        == "flashvsr.v2v.app:create_app"
    )
    assert "flashdreams-v2v" in manifest["project"]["dependencies"]
