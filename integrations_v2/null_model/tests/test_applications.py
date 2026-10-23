# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the NULL-model synthetic application entry points."""

from importlib.metadata import distribution

import pytest
from color_fade import ColorFadeApplication
from imgui_ui_demo.query_string_app import QueryStringApplication
from imgui_ui_demo.text_input_app import (
    TextInputApplication as ImGuiTextInputApplication,
)
from imgui_ui_demo.window_size_app import WindowSizeApplication
from red_screen import RedScreenApplication
from slangpy_ui_demo.invoke_async_app import ColorToggleApplication
from slangpy_ui_demo.model_output_app import ModelOutputApplication
from slangpy_ui_demo.text_input_app import (
    TextInputApplication as SlangPyTextInputApplication,
)

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2.session_desc import SessionDesc

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("name", "application_type"),
    [
        ("color-fade", ColorFadeApplication),
        ("red-screen", RedScreenApplication),
        ("imgui-ui-query-string", QueryStringApplication),
        ("imgui-ui-text-input", ImGuiTextInputApplication),
        ("imgui-ui-window-size", WindowSizeApplication),
        ("slangpy-ui-text-input", SlangPyTextInputApplication),
        ("slangpy-ui-model-output", ModelOutputApplication),
        ("slangpy-ui-invoke-async", ColorToggleApplication),
    ],
)
def test_entry_point_creates_standalone_application(
    name: str, application_type: type[IApplication]
) -> None:
    entry_points = distribution("flashdreams-null-model").entry_points
    (entry_point,) = entry_points.select(group="flashdreams.applications_v2", name=name)
    factory = entry_point.load()

    assert factory.__module__.startswith("null_model.apps.")
    app = factory()
    assert type(app) is application_type
    assert factory() is not app
    try:
        app.init([])
        assert isinstance(app.session_desc(), SessionDesc)
    finally:
        app.close()
