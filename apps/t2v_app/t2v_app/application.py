# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-to-video application runtime factory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from flashdreams.core.pipeline_presets import load_pipeline_provider
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams_runner import AppConfig, ApplicationArguments, Runtime

from .presets import PipelinePreset, RuntimePresetOptions, load_preset_catalog
from .runtime import T2VRuntime
from .session import T2VSessionDefaults

FIELD_PROMPT = "prompt"
FIELD_TOTAL_BLOCKS = "total_blocks"
FIELD_PIXEL_HEIGHT = "pixel_height"
FIELD_PIXEL_WIDTH = "pixel_width"
FIELD_FPS = "fps"


def create_runtime(arguments: ApplicationArguments) -> Runtime:
    """Parse T2V options and create an uninitialized application runtime.

    Args:
        arguments: Runner request containing the selected mode and parser.

    Returns:
        Runtime containing resolved pipeline and session configuration.
    """
    parser = arguments.parser
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
    options = vars(arguments.parse_args())

    preset_id, preset = _resolve_preset(options)
    scenario = _scenario(options, preset.runtime)
    total_steps = _total_steps(options, preset.runtime)
    return T2VRuntime(
        pipeline_config=_create_pipeline_config(preset_id, preset),
        session_defaults=T2VSessionDefaults(
            prompt=str(scenario[FIELD_PROMPT]),
            total_blocks=total_steps,
            pixel_height=_required_int(
                scenario[FIELD_PIXEL_HEIGHT], name=FIELD_PIXEL_HEIGHT
            ),
            pixel_width=_required_int(
                scenario[FIELD_PIXEL_WIDTH], name=FIELD_PIXEL_WIDTH
            ),
            fps=_required_int(scenario[FIELD_FPS], name=FIELD_FPS),
        ),
        config=AppConfig(
            model_id="t2v-app",
            fps=_required_int(scenario[FIELD_FPS], name=FIELD_FPS),
            output_layout=preset.runtime.output_layout,
            video_width=_required_int(
                scenario[FIELD_PIXEL_WIDTH], name=FIELD_PIXEL_WIDTH
            ),
            video_height=_required_int(
                scenario[FIELD_PIXEL_HEIGHT], name=FIELD_PIXEL_HEIGHT
            ),
            default_steps=total_steps,
        ),
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
    for name in (FIELD_PIXEL_HEIGHT, FIELD_PIXEL_WIDTH, FIELD_FPS):
        if _required_int(scenario[name], name=name) <= 0:
            raise ValueError(f"{name} must be > 0.")
    return scenario


def _total_steps(
    options: Mapping[str, object],
    defaults: RuntimePresetOptions,
) -> int | None:
    value = options.get(FIELD_TOTAL_BLOCKS)
    if value is None:
        value = defaults.total_blocks
    if value is None:
        return None
    total_steps = _required_int(value, name=FIELD_TOTAL_BLOCKS)
    if total_steps <= 0:
        raise ValueError(f"{FIELD_TOTAL_BLOCKS} must be > 0.")
    return total_steps


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


__all__ = ["create_runtime"]
