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

"""Cheap import-time checks for the ``causal_forcing`` plugin."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import pytest
import tomli as tomllib
from causal_forcing.config import (
    PIPELINE_CONFIGS,
    PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE,
)
from causal_forcing.t2v.app import create_app
from t2v import T2VApplication

pytestmark = pytest.mark.ci_gpu

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"
APPLICATION_SLUG = "t2v-causal-forcing"


def test_pipeline_configs_are_named_consistently() -> None:
    """Keep pipeline preset keys aligned with their public names."""
    assert PIPELINE_CONFIGS
    assert all(name == config.name for name, config in PIPELINE_CONFIGS.items())


def test_application_uses_chunkwise_pipeline_defaults() -> None:
    """Build the application directly from the default pipeline preset."""
    application = create_app()

    assert isinstance(application, T2VApplication)
    assert application.defaults.pipeline_config is PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE
    assert application.defaults.total_blocks == 60
    assert (application.defaults.pixel_height, application.defaults.pixel_width) == (
        480,
        832,
    )


def test_application_entry_point_matches_factory() -> None:
    """Keep the package manifest aligned with the application factory."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as file:
        metadata = tomllib.load(file)

    assert metadata["project"]["entry-points"][APPLICATION_ENTRY_POINT_GROUP] == {
        APPLICATION_SLUG: "causal_forcing.t2v.app:create_app"
    }


def test_application_entry_point_is_discoverable_when_installed() -> None:
    """Find the application slug when the plugin is installed."""
    discovered = {
        entry_point.name
        for entry_point in entry_points(group=APPLICATION_ENTRY_POINT_GROUP)
        if entry_point.value.startswith("causal_forcing.")
    }
    if not discovered:
        pytest.skip("plugin not installed; run `uv sync` from the repo root first")
    assert discovered == {APPLICATION_SLUG}
