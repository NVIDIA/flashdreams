# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video application definition for the FlashDreams app host."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flashdreams_app import (
    AppConfig,
    AppRequest,
    AppSpec,
    Mp4RunSpec,
    PipelineAppSpec,
    WebRTCRunSpec,
)

from flashdreams.core.pipeline_presets import load_pipeline_provider
from flashdreams.infra.decoder import StreamingVideoDecoder
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.runtime import InferenceInput

from .presets import PipelinePreset, RuntimePresetOptions, load_preset_catalog

FIELD_PROMPT = "prompt"
FIELD_TOTAL_BLOCKS = "total_blocks"
FIELD_PIXEL_HEIGHT = "pixel_height"
FIELD_PIXEL_WIDTH = "pixel_width"
FIELD_FPS = "fps"


def parse_options(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
) -> Mapping[str, Any]:
    """Parse T2V options with the selected mode's host parser.

    Args:
        parser: Parser preconfigured with host-owned presentation options.
        argv: Arguments remaining after the provider and mode.

    Returns:
        Parsed presentation and T2V options keyed by destination.
    """
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
    return vars(parser.parse_args(argv))


def create_app_spec(request: AppRequest) -> AppSpec:
    """Describe a T2V pipeline application without constructing its runtime.

    Args:
        request: Parsed host and T2V command-line values.

    Returns:
        Pipeline selection, initial conditioning, and presentation data.
    """
    options = request.options
    preset_id, preset = _resolve_preset(options)
    scenario = _scenario(options, preset.runtime)
    pipeline_config = _create_pipeline_config(preset_id, preset)
    return AppSpec(
        config=_build_app_config(scenario, preset.runtime),
        pipeline=PipelineAppSpec(
            pipeline_config=pipeline_config,
            initialize_cache=_initialize_cache,
        ),
        run=_build_run_spec(request.mode, options, scenario, preset.runtime),
    )


def _resolve_preset(
    options: Mapping[str, object],
) -> tuple[str, PipelinePreset[RuntimePresetOptions]]:
    """Resolve the configured pipeline preset."""
    catalog = load_preset_catalog(_optional_path(options.get("preset_config")))
    return catalog.resolve(_optional_string(options.get("preset_id")))


def _create_pipeline_config(
    preset_id: str,
    preset: PipelinePreset[RuntimePresetOptions],
) -> StreamInferencePipelineConfig:
    """Create the pipeline config selected by a resolved preset."""
    pipeline_provider = load_pipeline_provider(preset.provider)
    pipeline_config = pipeline_provider.create_pipeline_config(
        preset_id=preset_id,
        options=preset.pipeline,
    )
    if not isinstance(pipeline_config, StreamInferencePipelineConfig):
        raise TypeError(
            f"Pipeline provider returned {type(pipeline_config).__name__}, "
            "expected StreamInferencePipelineConfig."
        )
    if pipeline_config.name != preset_id:
        raise ValueError(
            f"Preset {preset_id!r} constructed pipeline "
            f"{pipeline_config.name!r}; the preset key and pipeline name must match."
        )
    return pipeline_config


def _build_app_config(
    scenario: Mapping[str, object],
    runtime_options: RuntimePresetOptions,
) -> AppConfig:
    """Build presentation configuration from the resolved T2V scenario."""
    return AppConfig(
        model_id="t2v-app",
        fps=_required_int(scenario[FIELD_FPS], name=FIELD_FPS),
        output_layout=runtime_options.output_layout,
        video_width=_required_int(scenario[FIELD_PIXEL_WIDTH], name=FIELD_PIXEL_WIDTH),
        video_height=_required_int(
            scenario[FIELD_PIXEL_HEIGHT], name=FIELD_PIXEL_HEIGHT
        ),
    )


def _build_run_spec(
    mode: str,
    options: Mapping[str, object],
    scenario: Mapping[str, object],
    defaults: RuntimePresetOptions,
) -> Mp4RunSpec | WebRTCRunSpec:
    """Build the selected presentation mode's session data."""
    initial_input = InferenceInput(
        global_conditioning={
            FIELD_PROMPT: scenario[FIELD_PROMPT],
            FIELD_PIXEL_HEIGHT: scenario[FIELD_PIXEL_HEIGHT],
            FIELD_PIXEL_WIDTH: scenario[FIELD_PIXEL_WIDTH],
        }
    )
    if mode == "webrtc":
        return WebRTCRunSpec(initial_input=initial_input)
    if mode == "mp4":
        total_steps = options.get(FIELD_TOTAL_BLOCKS)
        if total_steps is None:
            total_steps = defaults.total_blocks
        if total_steps is None:
            raise ValueError(
                "MP4 mode requires --total-blocks or runtime.total_blocks in "
                "the selected preset."
            )
        return Mp4RunSpec(
            initial_input=initial_input,
            total_steps=_required_int(total_steps, name=FIELD_TOTAL_BLOCKS),
        )
    raise ValueError(f"Unsupported presentation mode: {mode!r}.")


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
        FIELD_PIXEL_HEIGHT: _option_or_default(
            options, FIELD_PIXEL_HEIGHT, defaults.pixel_height
        ),
        FIELD_PIXEL_WIDTH: _option_or_default(
            options, FIELD_PIXEL_WIDTH, defaults.pixel_width
        ),
        FIELD_FPS: _option_or_default(options, FIELD_FPS, defaults.fps),
    }
    for name in (
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
        FIELD_PIXEL_HEIGHT,
        FIELD_PIXEL_WIDTH,
    )
    missing = tuple(name for name in required if name not in source)
    if missing:
        raise ValueError(f"Missing T2V global conditioning fields: {missing}.")
    scenario = dict(source)
    scenario[FIELD_PROMPT] = _resolve_prompt(scenario[FIELD_PROMPT])
    for name in (
        FIELD_PIXEL_HEIGHT,
        FIELD_PIXEL_WIDTH,
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


__all__ = ["create_app_spec", "parse_options"]
