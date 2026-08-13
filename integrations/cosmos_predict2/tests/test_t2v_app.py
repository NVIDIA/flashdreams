# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import tomli as tomllib
from cosmos_predict2.config import PIPELINE_COSMOS2_T2V_2B_720P
from cosmos_predict2.t2v.app import MODEL, createApp, create_app

from flashdreams.demo import Application, DemoAdapterApplication

pytestmark = pytest.mark.ci_cpu

APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"


def test_t2v_app_uses_default_pipeline_config() -> None:
    """The public app entry must remain owned by this integration package."""
    public_app = create_app()

    assert createApp is create_app
    assert isinstance(public_app, Application)
    assert isinstance(public_app, DemoAdapterApplication)
    assert MODEL.model_id == "cosmos-predict2-t2v"
    assert MODEL.preset_id == PIPELINE_COSMOS2_T2V_2B_720P.name
    assert MODEL.pipeline is PIPELINE_COSMOS2_T2V_2B_720P
    assert public_app.spec.model_id == MODEL.model_id
    assert public_app.spec.preset_id == MODEL.preset_id


def test_application_entry_point_matches_module_literal() -> None:
    """The integration owns its public T2V application entry point."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)

    entries = meta["project"]["entry-points"][APPLICATION_ENTRY_POINT_GROUP]
    assert entries == {
        "cosmos-predict2-t2v": "cosmos_predict2.t2v.app:create_app"
    }
