# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the version-sync pre-commit hook rewrites only package versions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu

_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "sync_version.py"
)


def test_sync_skips_test_harnesses_and_foreign_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location("sync_version", _SCRIPT)
    assert spec is not None and spec.loader is not None
    sync_version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync_version)
    monkeypatch.setattr(sync_version, "REPO_ROOT", tmp_path)

    package = tmp_path / "integrations_v2" / "demo" / "pyproject.toml"
    harness = (
        tmp_path / "integrations_v2" / "demo" / "tests" / "parity" / "pyproject.toml"
    )
    harness.parent.mkdir(parents=True)
    package.write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n\n'
        '[[tool.uv.dependency-metadata]]\nname = "upstream"\nversion = "0.1.0"\n'
    )
    harness.write_text('[project]\nname = "demo-parity"\nversion = "0.1.0"\n')

    assert sync_version.sync_version("9.9.9") == [package]
    assert package.read_text().count('version = "9.9.9"') == 1
    assert 'name = "upstream"\nversion = "0.1.0"' in package.read_text()
    assert 'version = "0.1.0"' in harness.read_text()
