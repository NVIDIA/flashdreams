# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from flashdreams.runtime.demo import DemoRoute
from interactive_drive_app import cli
from interactive_drive_app.application import DrivingSessionOutcome

pytestmark = pytest.mark.ci_cpu


def _adapter(model_id: str, *, compatible: bool) -> SimpleNamespace:
    routes = (
        (DemoRoute(input_mode="keyboard-driving", output_mode="local-window"),)
        if compatible
        else (DemoRoute(input_mode="replay", output_mode="mp4"),)
    )
    return SimpleNamespace(
        model_id=model_id,
        supported_routes=lambda: routes,
        list_sessions=lambda spec: (spec,),
    )


def test_list_models_only_prints_compatible_adapters(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "discover_demo_adapters",
        lambda: {
            "drive": _adapter("drive", compatible=True),
            "replay": _adapter("replay", compatible=False),
        },
    )

    cli.main(["--list-models"])

    assert capsys.readouterr().out == "drive\n"


def test_selected_adapter_runs_under_app_owned_lifecycle(monkeypatch) -> None:
    events: list[str] = []
    adapter = _adapter("drive", compatible=True)
    monkeypatch.setattr(cli, "discover_demo_adapters", lambda: {"drive": adapter})

    class _App:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["adapter"] is adapter
            events.append("init")

        def run_session(self, **kwargs: Any) -> DrivingSessionOutcome:
            events.append("run")
            return DrivingSessionOutcome(
                session_id=kwargs["session_id"],
                action="completed",
            )

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(cli, "InteractiveDriveApplication", _App)

    cli.main(["--model-id", "drive"])

    assert events == ["init", "run", "close"]


def test_incompatible_model_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "discover_demo_adapters",
        lambda: {"replay": _adapter("replay", compatible=False)},
    )

    with pytest.raises(SystemExit):
        cli.main(["--model-id", "replay"])
