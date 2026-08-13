# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the T2V application provider boundary."""

from __future__ import annotations

import argparse
from typing import Any, cast

import pytest
import t2v_app
from flashdreams_app import (
    AppProvider,
    AppRequest,
    AppSpec,
    Mp4RunSpec,
    PipelineAppSpec,
    WebRTCRunSpec,
)
from t2v_app import provider
from t2v_app.presets import (
    PipelinePreset,
    PresetCatalog,
    RuntimePresetOptions,
)

from flashdreams.infra.pipeline import StreamInferencePipelineConfig

pytestmark = pytest.mark.ci_cpu


def test_provider_module_conforms_to_host_contract() -> None:
    assert isinstance(t2v_app, AppProvider)


def test_t2v_provider_parses_model_options() -> None:
    parser = argparse.ArgumentParser()
    options = provider.parse_options(parser, ["--prompt", "A waterfall"])
    assert options["prompt"] == "A waterfall"
    assert options["preset_id"] is None
    assert options["preset_config"] is None
    assert "backend" not in options
    assert "compile" not in options


def test_create_app_spec_returns_data_without_constructing_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_constructed = False

    class Pipeline:
        def __init__(self, _: object) -> None:
            nonlocal pipeline_constructed
            pipeline_constructed = True

    preset_id = "test-t2v"
    defaults = RuntimePresetOptions(
        prompt="default prompt",
        total_blocks=2,
        pixel_height=64,
        pixel_width=96,
        fps=12,
        output_layout="tchw",
    )
    preset = PipelinePreset(
        provider="tests:provider",
        runtime=defaults,
        pipeline={},
    )
    catalog = PresetCatalog(
        default_preset_id=preset_id,
        presets={preset_id: preset},
    )
    pipeline_config = StreamInferencePipelineConfig(
        _target=cast(Any, Pipeline),
        name=preset_id,
        diffusion_model=cast(Any, None),
    )

    class Provider:
        def create_pipeline_config(
            self, *, preset_id: str, options: object
        ) -> StreamInferencePipelineConfig:
            assert preset_id == "test-t2v"
            assert options == {}
            return pipeline_config

    monkeypatch.setattr(provider, "load_preset_catalog", lambda _: catalog)
    monkeypatch.setattr(provider, "load_pipeline_provider", lambda _: Provider())

    created = provider.create_app_spec(
        AppRequest(
            mode="mp4",
            options={
                "preset_config": None,
                "preset_id": None,
                "prompt": "A waterfall",
                "total_blocks": None,
                "pixel_height": None,
                "pixel_width": None,
                "fps": None,
            },
        )
    )

    assert isinstance(created, AppSpec)
    assert isinstance(created.pipeline, PipelineAppSpec)
    assert created.pipeline.pipeline_config is pipeline_config
    assert created.config.video_width == 96
    assert created.config.fps == 12
    assert isinstance(created.run, Mp4RunSpec)
    assert created.run.initial_input.global_conditioning["prompt"] == "A waterfall"
    assert created.run.total_steps == 2
    assert not pipeline_constructed


def test_webrtc_run_spec_does_not_require_total_steps() -> None:
    defaults = RuntimePresetOptions(
        prompt="default prompt",
        pixel_height=64,
        pixel_width=96,
        fps=12,
        output_layout="tchw",
    )
    scenario = {
        provider.FIELD_PROMPT: "A waterfall",
        provider.FIELD_PIXEL_HEIGHT: 64,
        provider.FIELD_PIXEL_WIDTH: 96,
        provider.FIELD_FPS: 12,
    }

    run_spec = provider._build_run_spec(
        "webrtc",
        {"total_blocks": None},
        scenario,
        defaults,
    )

    assert isinstance(run_spec, WebRTCRunSpec)
    assert run_spec.initial_input.global_conditioning["prompt"] == "A waterfall"
