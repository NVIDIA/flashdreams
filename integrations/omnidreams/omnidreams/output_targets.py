# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams output capabilities for ``flashdreams-run``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.output_targets import (
    OutputLaunchOptions,
    OutputMode,
    OutputTargetSpec,
)

_LOCAL_WINDOW_MANIFESTS = {
    "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae": "example_world_model.yaml",
    "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf": (
        "example_world_model_perf.yaml"
    ),
    "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-native-perf": (
        "example_world_model_perf.yaml"
    ),
}


class OmnidreamsOutputTargetAdapter:
    def supported_modes(
        self,
        config: RunnerConfig,
        options: OutputLaunchOptions,
    ) -> tuple[OutputMode, ...]:
        modes: list[OutputMode] = []
        if _is_single_view(config):
            modes.append("webrtc")
        if _local_window_manifest(config, options) is not None:
            modes.append("local-window")
        return tuple(modes)

    def resolve(
        self,
        config: RunnerConfig,
        *,
        mode: OutputMode,
        options: OutputLaunchOptions,
    ) -> OutputTargetSpec | None:
        if mode == "webrtc" and _is_single_view(config):
            return _webrtc_spec(config, options)
        if mode == "local-window":
            manifest = _local_window_manifest(config, options)
            if manifest is not None:
                return _local_window_spec(config, manifest)
        return None


def _webrtc_spec(
    config: RunnerConfig,
    options: OutputLaunchOptions,
) -> OutputTargetSpec:
    argv = [
        "webrtc",
        "--preset-id",
        _pipeline_name(config),
        "--device",
        str(config.device),
        "--fps",
        str(getattr(config, "output_fps", 30)),
        "--video-height",
        str(getattr(config, "pixel_height", 704)),
        "--video-width",
        str(getattr(config, "pixel_width", 1280)),
    ]
    seed = _diffusion_seed(config)
    if seed is not None:
        argv.extend(("--seed", str(seed)))
    _append_postprocess_preset(argv, config)
    if options.host:
        argv.extend(("--host", options.host))
    if options.port is not None:
        argv.extend(("--port", str(options.port)))
    if options.prefer_sw_encoder:
        argv.append("--prefer-sw-encoder")
    return OutputTargetSpec(
        mode="webrtc",
        label="OmniDreams shared demo WebRTC server",
        module="omnidreams.demo.app",
        argv=tuple(argv),
    )


def _local_window_spec(config: RunnerConfig, manifest: Path) -> OutputTargetSpec:
    argv = ["--manifest", str(manifest)]
    _append_postprocess_preset(argv, config)
    return OutputTargetSpec(
        mode="local-window",
        label="Omnidreams local interactive window",
        module="omnidreams.interactive_drive",
        argv=tuple(argv),
        notes=(
            "Local-window uses the OmniDreams interactive-drive manifest for "
            "scene, resolution, and runtime-specific controls.",
        ),
    )


def _local_window_manifest(
    config: RunnerConfig,
    options: OutputLaunchOptions,
) -> Path | None:
    if options.local_window_manifest is not None:
        return options.local_window_manifest
    manifest = _LOCAL_WINDOW_MANIFESTS.get(config.runner_name)
    return None if manifest is None else Path(manifest)


def _pipeline_name(config: RunnerConfig) -> str:
    name = getattr(config.pipeline, "name", None)
    return str(name or config.runner_name)


def _diffusion_seed(config: RunnerConfig) -> int | None:
    diffusion_model = getattr(config.pipeline, "diffusion_model", None)
    seed = getattr(diffusion_model, "seed", None)
    return None if seed is None else int(seed)


def _is_single_view(config: RunnerConfig) -> bool:
    diffusion_model = getattr(config.pipeline, "diffusion_model", None)
    transformer: Any = getattr(diffusion_model, "transformer", None)
    return int(getattr(transformer, "num_views", 1)) == 1


def _append_postprocess_preset(argv: list[str], config: RunnerConfig) -> None:
    preset = config.postprocess.preset
    if preset:
        argv.extend(("--postprocess-preset", str(preset)))


OUTPUT_TARGET_ADAPTER = OmnidreamsOutputTargetAdapter()

__all__ = ["OUTPUT_TARGET_ADAPTER", "OmnidreamsOutputTargetAdapter"]
