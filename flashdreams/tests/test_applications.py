# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInputSchema,
)
from flashdreams.serving import application_launcher

pytestmark = pytest.mark.ci_cpu


class _Application:
    application_name = "example"
    description = "Example application"
    model_id = "example-model"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()
    config = InferenceConfig(model_id=model_id, device="cpu")
    scenario = None
    fps = 30
    video_width = 16
    video_height = 16
    output_layout = "tchw"
    default_mode = "null"
    title = None
    supported_control_keys = frozenset()

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay", "keyboard-driving")

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4", "null", "webrtc", "local-window")

    def default_input_mapping(self) -> IdentityInputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        del config

    def create_runtime(self, config: InferenceConfig) -> Any:
        del config
        raise AssertionError("runtime should not be created")

    def prepare_scenario(self, spec: Any) -> Any:
        del spec
        raise AssertionError("scenario should not be prepared")

    def create_model_input_provider(self, spec: Any, scenario: Any) -> Any:
        del spec, scenario
        raise AssertionError("provider should not be created")


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    value: object

    def load(self) -> object:
        return self.value


def test_application_entry_point_receives_remaining_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    factory_args: list[list[str]] = []
    run_args: list[dict[str, object]] = []

    def factory(args: list[str]) -> _Application:
        factory_args.append(args)
        return _Application()

    monkeypatch.setattr(
        application_launcher,
        "entry_points",
        lambda **_kwargs: [_EntryPoint("example", factory)],
    )
    application_launcher.application_factories.cache_clear()
    monkeypatch.setattr(
        application_launcher,
        "_run_application",
        lambda application, **kwargs: run_args.append(
            {"application": application, **kwargs}
        ),
    )
    output = tmp_path / "result.mp4"

    handled = application_launcher.run_application_from_argv(
        [
            "example",
            "mp4",
            "--output",
            str(output),
            "--model-flag",
            "value",
        ]
    )

    assert handled
    assert factory_args == [["--model-flag", "value"]]
    assert run_args[0]["mode"] == "mp4"
    assert run_args[0]["output_path"] == output


def test_unknown_application_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application_launcher, "entry_points", lambda **_kwargs: [])
    application_launcher.application_factories.cache_clear()

    assert not application_launcher.run_application_from_argv(["missing"])
