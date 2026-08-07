# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

import interactive_drive_app.application as application_module
import interactive_drive_app.runtime as runtime_module
import pytest
from flashdreams.runtime import DRIVER_COMMAND, InputMappingSchema
from flashdreams.runtime.demo import DemoAdapter, DemoSpec, LocalWindowOutputSpec
from flashdreams.serving.presentation import KeyEvent
from interactive_drive_app.application import DrivingSessionOutcome

pytestmark = pytest.mark.ci_cpu


def test_run_driving_session_owns_application_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    ended = DrivingSessionOutcome(session_id="session-0", action="completed")

    class _FakeApplication:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["initial_spec"].input_mode == "keyboard-driving"
            events.append("init")

        def run_session(self, **kwargs: Any) -> DrivingSessionOutcome:
            assert kwargs["session_id"] == "session-0"
            events.append("run")
            return ended

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(runtime_module, "InteractiveDriveApplication", _FakeApplication)
    spec = DemoSpec(
        model_id="compatible-model",
        input_mode="keyboard-driving",
        output=LocalWindowOutputSpec(),
    )

    result = runtime_module.run_driving_session(
        spec=spec,
        adapter=cast(DemoAdapter, object()),
    )

    assert result is ended
    assert events == ["init", "run", "close"]


def test_driving_mapping_must_consume_driver_command() -> None:
    class _Mapping:
        mapping_schema = InputMappingSchema(name="ignores-driving")

    with pytest.raises(ValueError, match="must consume driver_command"):
        application_module._require_driver_command_mapping(
            cast(Any, _Mapping()),
        )


def test_declared_driver_command_mapping_is_accepted() -> None:
    class _Mapping:
        mapping_schema = InputMappingSchema(
            name="driving",
            consumes=(DRIVER_COMMAND,),
        )

    application_module._require_driver_command_mapping(
        cast(Any, _Mapping()),
    )


@pytest.mark.parametrize(
    ("key", "action"),
    (
        ("r", "reset"),
        ("x", "exit"),
        ("tab", "next"),
        ("backspace", "previous"),
    ),
)
def test_session_control_keys_stay_out_of_model_input(key: str, action: str) -> None:
    controls = application_module._SessionControls()
    overlay = application_module._SessionControlOverlay(controls)

    assert overlay.on_key(KeyEvent(key=key, action="press", timestamp_s=0.0))
    assert controls.consume() == action
