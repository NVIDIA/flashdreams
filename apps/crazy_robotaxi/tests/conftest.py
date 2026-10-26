# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated user storage and authored application map fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_user_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def race_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from crazy_robotaxi import application

    source = Path(__file__).parent / "maps" / "race_course.robotaxi.yaml"
    catalog = tmp_path / "maps"
    catalog.mkdir()
    path = catalog / source.name
    path.write_bytes(source.read_bytes())
    # Discovery scans the default map directory even when --map is explicit.
    monkeypatch.setattr(application, "_DEFAULT_MAP", path)
    return path
