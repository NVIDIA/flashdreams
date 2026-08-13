# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video application definition for the FlashDreams app host."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flashdreams_app import (
    AppConfig,
    PipelineAppSpec,
    PipelineContract,
    RuntimeMetadata,
    require_pipeline_config,
)

from flashdreams.core.pipeline_presets import load_pipeline_provider
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.runtime import InferenceInput

from .presets import RuntimePresetOptions, load_preset_catalog

FIELD_PROMPT = "prompt"
FIELD_TOTAL_BLOCKS = "total_blocks"
FIELD_PIXEL_HEIGHT = "pixel_height"
FIELD_PIXEL_WIDTH = "pixel_width"
FIELD_FPS = "fps"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register T2V conditioning and rollout options on the host parser."""
    parser.add_argument(
        "--preset-config",
        type=Path,
        help="Pipeline preset YAML (defaults to t2v_app's packaged catalog)",
    )
    parser.add_argument(
        "--preset-id",
        help="Preset key (defaults to default_preset_id from the YAML)",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--total-blocks", type=int)
    parser.add_argument("--height", type=int, dest=FIELD_PIXEL_HEIGHT)
    parser.add_argument("--width", type=int, dest=FIELD_PIXEL_WIDTH)
    parser.add_argument("--fps", type=int)


def create_app_spec(config: AppConfig) -> PipelineAppSpec:
    """Describe a T2V pipeline application without constructing its runtime."""
    options = config.options
    catalog = load_preset_catalog(_optional_path(options.get("preset_config")))
    preset_id, preset = catalog.resolve(_optional_string(options.get("preset_id")))
    provider = load_pipeline_provider(preset.provider)
    pipeline_config = require_pipeline_config(
        provider.create_pipeline_config(
            preset_id=preset_id,
            options=preset.pipeline,
        ),
        expected_name=preset_id,
    )
    scenario = _scenario(options, preset.runtime)
    return PipelineAppSpec(
        pipeline_config=pipeline_config,
        contract=PipelineContract(initialize_cache=_initialize_cache),
        metadata=RuntimeMetadata(
            model_id="t2v-app",
            fps=_required_int(scenario[FIELD_FPS], name=FIELD_FPS),
            output_layout=preset.runtime.output_layout,
            video_width=_required_int(
                scenario[FIELD_PIXEL_WIDTH], name=FIELD_PIXEL_WIDTH
            ),
            video_height=_required_int(
                scenario[FIELD_PIXEL_HEIGHT], name=FIELD_PIXEL_HEIGHT
            ),
        ),
        initial_input=InferenceInput(global_conditioning=scenario),
        total_steps=_required_int(
            scenario[FIELD_TOTAL_BLOCKS], name=FIELD_TOTAL_BLOCKS
        ),
        result_metadata={FIELD_PROMPT: scenario[FIELD_PROMPT]},
    )


def _initialize_cache(pipeline: Any, inputs: InferenceInput) -> object:
    """Bind T2V prompt and dimensions to a new pipeline cache."""
    scenario = _scenario_from_inputs(inputs)
    decoder = pipeline.decoder
    if not isinstance(decoder, StreamingVideoDecoder):
        raise TypeError("T2V pipelines require a StreamingVideoDecoder.")
    ratio = decoder.spatial_compression_ratio
    pixel_height = _required_int(scenario[FIELD_PIXEL_HEIGHT], name=FIELD_PIXEL_HEIGHT)
    pixel_width = _required_int(scenario[FIELD_PIXEL_WIDTH], name=FIELD_PIXEL_WIDTH)
    if pixel_height % ratio or pixel_width % ratio:
        raise ValueError(
            "T2V dimensions must be divisible by the decoder spatial "
            f"compression ratio ({ratio})."
        )
    return pipeline.initialize_cache(
        text=[str(scenario[FIELD_PROMPT])],
        image=None,
        height=pixel_height // ratio,
        width=pixel_width // ratio,
    )


def _scenario(
    options: Mapping[str, object],
    defaults: RuntimePresetOptions,
) -> dict[str, object]:
    prompt_value = options.get(FIELD_PROMPT)
    prompt = _resolve_prompt(defaults.prompt if prompt_value is None else prompt_value)
    scenario = {
        FIELD_PROMPT: prompt,
        FIELD_TOTAL_BLOCKS: _option_or_default(
            options, FIELD_TOTAL_BLOCKS, defaults.total_blocks
        ),
        FIELD_PIXEL_HEIGHT: _option_or_default(
            options, FIELD_PIXEL_HEIGHT, defaults.pixel_height
        ),
        FIELD_PIXEL_WIDTH: _option_or_default(
            options, FIELD_PIXEL_WIDTH, defaults.pixel_width
        ),
        FIELD_FPS: _option_or_default(options, FIELD_FPS, defaults.fps),
    }
    for name in (
        FIELD_TOTAL_BLOCKS,
        FIELD_PIXEL_HEIGHT,
        FIELD_PIXEL_WIDTH,
        FIELD_FPS,
    ):
        if _required_int(scenario[name], name=name) <= 0:
            raise ValueError(f"{name} must be > 0.")
    return scenario


def _scenario_from_inputs(inputs: InferenceInput) -> Mapping[str, object]:
    source = inputs.global_conditioning
    required = (
        FIELD_PROMPT,
        FIELD_TOTAL_BLOCKS,
        FIELD_PIXEL_HEIGHT,
        FIELD_PIXEL_WIDTH,
        FIELD_FPS,
    )
    missing = tuple(name for name in required if name not in source)
    if missing:
        raise ValueError(f"Missing T2V global conditioning fields: {missing}.")
    scenario = dict(source)
    scenario[FIELD_PROMPT] = _resolve_prompt(scenario[FIELD_PROMPT])
    for name in (
        FIELD_TOTAL_BLOCKS,
        FIELD_PIXEL_HEIGHT,
        FIELD_PIXEL_WIDTH,
        FIELD_FPS,
    ):
        if _required_int(scenario[name], name=name) <= 0:
            raise ValueError(f"{name} must be > 0.")
    return scenario


def _resolve_prompt(value: object) -> str:
    if isinstance(value, Path):
        lines = (line.strip() for line in value.read_text().splitlines())
        prompt = next((line for line in lines if line), "")
    else:
        prompt = str(value).strip()
    if not prompt:
        raise ValueError("A non-empty text-to-video prompt is required.")
    return prompt


def _option_or_default(
    options: Mapping[str, object], name: str, default: object
) -> object:
    value = options.get(name)
    return default if value is None else value


def _required_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be integer-compatible, got {value!r}.")
    return int(value)


def _optional_path(value: object) -> str | Path | None:
    if value is None or isinstance(value, (str, Path)):
        return value
    raise TypeError(f"Expected path or None, got {type(value).__name__}.")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError("preset_id must be a non-empty string or None.")
    return value.strip()


__all__ = ["add_arguments", "create_app_spec"]
