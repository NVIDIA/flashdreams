# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership tests for v2 applications and model adapters."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import tomli as tomllib

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).resolve().parents[3]

_APPLICATIONS = (
    "action2v",
    "color_fade",
    "hdmap2v",
    "interactive_drive",
    "red_screen",
    "slangpy_ui_demo",
)
"""Complete application packages that must remain under ``apps``."""

_MODEL_ADAPTERS = (
    ("cosmos_predict2", "t2v", "t2v-cosmos-predict2"),
    ("causal_forcing", "t2v", "t2v-causal-forcing"),
    ("fastvideo_causal_wan22", "t2v", "t2v-fastvideo-causal-wan22"),
    ("self_forcing", "t2v", "t2v-self-forcing"),
    ("wan21", "t2v", "t2v-wan21"),
    ("wan22", "ti2v", "ti2v-wan22"),
    ("lingbot", "action2v", "action2v-lingbot"),
    ("hy_worldplay", "action2v", "action2v-hy-worldplay"),
    ("omnidreams", "hdmap2v", "hdmap2v"),
    ("omnidreams", "interactive_drive", "interactive-drive"),
)
"""Model directory, app adapter, and application entry-point slug tuples."""


def test_t2v_framework_installs_slangpy_and_serving_support() -> None:
    manifest = tomllib.loads((_REPO_ROOT / "apps/t2v/pyproject.toml").read_text())
    assert "flashdreams[local-window,serving]" in manifest["project"]["dependencies"]


def test_every_model_project_uses_the_v2_package_layout() -> None:
    """Keep the config, implementation, app bindings, and tests separated."""
    legacy_root = _REPO_ROOT / "integrations"
    assert not legacy_root.exists() or not any(legacy_root.iterdir())

    standard_root_items = {
        "README.md",
        "apps",
        "config.py",
        "impl",
        "pyproject.toml",
        "tests",
    }
    important_scripts = {
        "hy_worldplay": {"run-docker.sh"},
        "omnidreams": {"compile_protos.sh"},
    }

    for project_dir in (_REPO_ROOT / "integrations_v2").iterdir():
        if not project_dir.is_dir() or not (project_dir / "pyproject.toml").is_file():
            continue
        assert (project_dir / "config.py").is_file(), project_dir.name
        assert sorted(path.name for path in project_dir.glob("*.py")) == ["config.py"]
        assert not (project_dir / project_dir.name).exists(), project_dir.name
        source_items = {
            path.name
            for path in project_dir.iterdir()
            if path.name != "__pycache__" and not path.name.endswith(".egg-info")
        }
        allowed_items = standard_root_items | important_scripts.get(
            project_dir.name, set()
        )
        assert source_items <= allowed_items, (
            project_dir.name,
            sorted(source_items - allowed_items),
        )


def test_capability_apps_are_separate_from_model_bindings() -> None:
    """Keep application implementations out of model packages."""
    assert (_REPO_ROOT / "integrations_v2/lingbot/apps/action2v/adapter.py").is_file()
    assert (_REPO_ROOT / "integrations_v2/hy_worldplay/apps/action2v/adapter.py").is_file()
    assert not (_REPO_ROOT / "integrations_v2/lingbot/webrtc").exists()
    assert (_REPO_ROOT / "integrations_v2/omnidreams/apps/hdmap2v/adapter.py").is_file()
    assert (
        _REPO_ROOT / "integrations_v2/omnidreams/apps/interactive_drive/adapter.py"
    ).is_file()
    assert not (_REPO_ROOT / "apps/lingbot_demo").exists()
    assert not (_REPO_ROOT / "apps/omnidreams_demo").exists()
    assert not list((_REPO_ROOT / "apps/action2v/action2v").glob("tests/**/*.py"))
    assert not list((_REPO_ROOT / "apps/action2v/action2v").glob("webrtc/**/*.py"))
    assert not list((_REPO_ROOT / "apps/hdmap2v").glob("**/tests/**/*.py"))
    assert (
        _REPO_ROOT / "apps/hdmap2v/hdmap2v/interactive_drive"
    ).is_dir()


@pytest.mark.parametrize("project", _APPLICATIONS)
def test_application_packages_are_owned_by_apps(project: str) -> None:
    """Keep complete application implementations out of ``integrations_v2``."""
    assert (_REPO_ROOT / "apps" / project / "pyproject.toml").is_file()
    assert not (_REPO_ROOT / "integrations_v2" / project).exists()


@pytest.mark.parametrize(("model", "app_name", "slug"), _MODEL_ADAPTERS)
def test_model_adapter_only_loads_config_for_shared_app(
    model: str, app_name: str, slug: str
) -> None:
    """Reject application implementations in model-owned adapters."""
    project_dir = _REPO_ROOT / "integrations_v2" / model
    manifest = tomllib.loads((project_dir / "pyproject.toml").read_text())
    adapter_module = project_dir / "apps" / app_name / "adapter.py"
    tree = ast.parse(adapter_module.read_text())
    config_imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "config"
    ]

    assert (project_dir / "config.py").is_file()
    assert not (_REPO_ROOT / "integrations" / model).exists()
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    assert len(config_imports) == 1
    assert config_imports[0].level == 3
    assert manifest["project"]["entry-points"]["flashdreams.applications_v2"][slug] == (
        f"{model}.apps.{app_name}.adapter:create_app"
    )


def test_integrations_do_not_ship_legacy_runner_or_launch_modules() -> None:
    """Keep application execution and launch orchestration under ``apps``."""
    forbidden = {"runner.py", "launch.py", "runtime.py", "prepare.py", "model_session.py"}
    violations = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in (_REPO_ROOT / "integrations_v2").glob("*/*.py")
        if path.name in forbidden
    ]
    assert violations == []
