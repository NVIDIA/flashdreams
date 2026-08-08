# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot output capabilities for ``flashdreams-run``."""

from __future__ import annotations

from typing import Any

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.output_targets import (
    OutputLaunchOptions,
    OutputMode,
    OutputTargetSpec,
)


class LingbotOutputTargetAdapter:
    def supported_modes(
        self,
        config: RunnerConfig,
        options: OutputLaunchOptions,
    ) -> tuple[OutputMode, ...]:
        del config, options
        return ("webrtc",)

    def resolve(
        self,
        config: RunnerConfig,
        *,
        mode: OutputMode,
        options: OutputLaunchOptions,
    ) -> OutputTargetSpec | None:
        if mode != "webrtc":
            return None
        argv = [
            "webrtc",
            "--preset-id",
            _pipeline_name(config),
            "--device",
            str(config.device),
            "--fps",
            str(getattr(config, "fps", 16)),
            "--video-height",
            str(getattr(config, "pixel_height", 464)),
            "--video-width",
            str(getattr(config, "pixel_width", 832)),
        ]
        if _compile_network(config) is False:
            argv.append("--no-compile")
        example_idx = getattr(config, "example_idx", None)
        if example_idx is not None:
            argv.extend(("--example-idx", str(example_idx)))
        if options.host:
            argv.extend(("--host", options.host))
        if options.port is not None:
            argv.extend(("--port", str(options.port)))
        if options.prefer_sw_encoder:
            argv.append("--prefer-sw-encoder")
        return OutputTargetSpec(
            mode="webrtc",
            label="LingBot shared demo WebRTC server",
            module="lingbot.demo.app",
            argv=tuple(argv),
        )


def _pipeline_name(config: RunnerConfig) -> str:
    name = getattr(config.pipeline, "name", None)
    return str(name or config.runner_name)


def _compile_network(config: RunnerConfig) -> bool | None:
    diffusion_model = getattr(config.pipeline, "diffusion_model", None)
    transformer: Any = getattr(diffusion_model, "transformer", None)
    value = getattr(transformer, "compile_network", None)
    return None if value is None else bool(value)


OUTPUT_TARGET_ADAPTER = LingbotOutputTargetAdapter()

__all__ = ["OUTPUT_TARGET_ADAPTER", "LingbotOutputTargetAdapter"]
