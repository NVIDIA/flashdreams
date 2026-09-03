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


@pytest.mark.parametrize(("model", "factories"), _MODEL_APPLICATIONS)
def test_model_owns_v2_t2v_adapter(model: str, factories: dict[str, str]) -> None:
    """Keep model code and its T2V adapter in one v2 integration package."""
    project = _REPO_ROOT / "integrations_v2" / model
    application_project = project / "apps" / "t2v"
    implementation = project / "impl"
    config = project / "config.py"
    manifest = tomllib.loads((project / "pyproject.toml").read_text())

    assert not (_REPO_ROOT / "integrations" / model).exists()
    assert {path.name for path in project.glob("*.py")} == {"__init__.py", "config.py"}
    assert not (project / "runner.py").exists()
    assert {path.name for path in application_project.glob("*.py")} == {
        "__init__.py",
        "adapter.py",
    }
    assert not (application_project / "t2v").exists()
    assert config.is_file()
    assert (implementation / "__init__.py").is_file()
    assert not (implementation / "config.py").exists()
    for source_path in implementation.rglob("*.py"):
        source = source_path.read_text()
        assert "from t2v" not in source
        assert "import t2v" not in source
        assert "flashdreams.runtime_v2" not in source
        assert "flashdreams.serving" not in source
    assert not list(project.glob("apps/*/config.py"))
    application_readme = (application_project / "README.md").read_text()
    model_readme = (project / "README.md").read_text()
    for slug in factories:
        assert slug in application_readme
        assert slug in model_readme
    assert "../../../../apps/t2v/README.md" in application_readme
    assert not (application_project / "tests").exists()
    assert not (application_project / "impl").exists()
    assert (project / "tests").is_dir()
    assert "flashdreams-t2v" in manifest["project"]["dependencies"]
    assert f"{model}.impl" in manifest["tool"]["setuptools"]["packages"]
    for slug, factory in factories.items():
        assert factory == "create_app" or (
            factory.startswith("create_app_")
            and slug.endswith(
                "-" + factory.removeprefix("create_app_").replace("_", "-")
            )
        )
    assert manifest["tool"]["setuptools"]["package-dir"] == {model: "."}
    assert manifest["project"]["entry-points"]["flashdreams.applications_v2"] == {
        slug: f"{model}.apps.t2v.adapter:{factory}"
        for slug, factory in factories.items()
    }
    assert "flashdreams.runner_configs" not in manifest["project"]["entry-points"]
    assert "RUNNER_" not in config.read_text()


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
