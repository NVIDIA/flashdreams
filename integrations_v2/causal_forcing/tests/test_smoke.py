# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU configuration checks for the causal_forcing v2 application binding."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli as tomllib
from causal_forcing.config import T2V_APPLICATION_DEFAULTS

pytestmark = pytest.mark.ci_cpu


def test_application_defaults_reference_the_model_pipeline() -> None:
    """Keep the application binding pointed at a named model config."""
    assert T2V_APPLICATION_DEFAULTS.pipeline_config.name
    assert T2V_APPLICATION_DEFAULTS.prompt.strip()
    assert T2V_APPLICATION_DEFAULTS.total_blocks > 0


def test_application_entry_point_uses_the_nested_adapter() -> None:
    """Keep application discovery on the model-owned nested adapter."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as stream:
        metadata = tomllib.load(stream)
    entries = metadata["project"]["entry-points"]["flashdreams.applications_v2"]
    assert entries == {"t2v-causal-forcing": "causal_forcing.apps.t2v.adapter:create_app"}
