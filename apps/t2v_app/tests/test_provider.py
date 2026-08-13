# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the T2V application provider boundary."""

from __future__ import annotations

import argparse
from typing import Any, cast

import pytest
import t2v_app
from flashdreams_app import AppConfig, AppProvider, PipelineAppSpec
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


def test_t2v_provider_registers_model_options() -> None:
    parser = argparse.ArgumentParser()
    provider.add_arguments(parser)
    args = parser.parse_args(["--prompt", "A waterfall"])
    assert args.prompt == "A waterfall"
    assert args.preset_id is None
    assert args.preset_config is None
    assert not hasattr(args, "backend")
    assert not hasattr(args, "compile")


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
        AppConfig(
            options={
                "preset_config": None,
                "preset_id": None,
                "prompt": "A waterfall",
                "total_blocks": None,
                "pixel_height": None,
                "pixel_width": None,
                "fps": None,
            }
        )
    )

    assert isinstance(created, PipelineAppSpec)
    assert created.pipeline_config is pipeline_config
    assert created.initial_input.global_conditioning["prompt"] == "A waterfall"
    assert created.metadata.video_width == 96
    assert created.metadata.fps == 12
    assert created.total_steps == 2
    assert not pipeline_constructed
