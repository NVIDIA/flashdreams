# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-layout checks for T2V application ownership."""

from pathlib import Path

import pytest
import tomli as tomllib

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).resolve().parents[3]

_MODEL_APPLICATIONS = (
    (
        "causal_forcing",
        {
            "t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise": "create_app",
            "t2v-causal-forcing-wan2.1-t2v-1.3b-framewise": "create_app_framewise",
        },
    ),
    (
        "cosmos_predict2",
        {"t2v-cosmos2-t2v-2b-720p": "create_app"},
    ),
    (
        "fastvideo_causal_wan22",
        {"t2v-fastvideo-causal-wan2.2-t2v-14b": "create_app"},
    ),
    (
        "self_forcing",
        {
            "t2v-self-forcing-wan2.1-t2v-1.3b": "create_app",
            "t2v-self-forcing-wan2.1-t2v-1.3b-taehv": "create_app_taehv",
            "t2v-self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope": (
                "create_app_sink5_window7_rerope"
            ),
        },
    ),
    ("wan21", {"t2v-wan21-t2v-1.3b-480p": "create_app"}),
    ("wan22", {"t2v-wan22-ti2v-5b": "create_app"}),
)
"""Model packages and their user-facing application slugs."""


def test_shared_t2v_code_is_owned_by_app_package() -> None:
    """Keep reusable demo infrastructure outside the framework and models."""
    project = _REPO_ROOT / "apps" / "t2v"
    package = project / "t2v"
    manifest = tomllib.loads((project / "pyproject.toml").read_text())

    assert {path.name for path in project.glob("*.py")} == set()
    assert {path.name for path in package.glob("*.py")} == {
        "__init__.py",
        "application.py",
        "defaults.py",
        "session.py",
        "testing.py",
        "ui.py",
    }
    assert (project / "tests").is_dir()
    readme = (project / "README.md").read_text()
    for heading in ("## Controls", "## Usage", "## Tests"):
        assert heading in readme
    assert manifest["tool"]["setuptools"]["packages"]["find"]["include"] == ["t2v*"]
    assert not (_REPO_ROOT / "flashdreams" / "flashdreams" / "t2v_v2").exists()
    for model, _factories in _MODEL_APPLICATIONS:
        assert not (_REPO_ROOT / "integrations_v2" / f"t2v_{model}").exists()
