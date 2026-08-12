# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LingBot launch capability for ``flashdreams-run``."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Literal, cast

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.launch import (
    LaunchMode,
    LaunchOptions,
    ResolvedLaunch,
)

_REPLAY_SCENARIO_FIELDS = frozenset(
    {
        "prompt",
        "prompt_path",
        "image_path",
        "pose_path",
        "intrinsic_path",
        "example_data",
        "example_idx",
        "total_blocks",
        "pixel_height",
        "pixel_width",
        "fps",
    }
)
_WEBRTC_SCENARIO_FIELDS = frozenset({"example_idx"})
_WEBRTC_OUTPUT_FIELDS = frozenset(
    {
        "host",
        "port",
        "seed",
        "fps",
        "video_height",
        "video_width",
        "warmup_chunks",
        "warmup_timeout_s",
        "client_liveness_timeout_s",
        "prefer_sw_encoder",
    }
)


class LingbotLaunchCapability:
    """Construct LingBot replay and WebRTC launches directly."""

    def supported_modes(
        self,
        config: RunnerConfig,
        options: LaunchOptions,
    ) -> tuple[LaunchMode, ...]:
        del config, options
        return ("mp4", "null", "webrtc")

    def resolve(
        self,
        config: RunnerConfig,
        *,
        mode: LaunchMode,
        options: LaunchOptions,
    ) -> ResolvedLaunch | None:
        if mode in {"mp4", "null"}:
            _validate_fields("scenario", options.scenario, _REPLAY_SCENARIO_FIELDS)
            _validate_fields("output", options.output, {"path", "output", "fps"})
            output_path = options.output.get("path") or options.output.get("output")
            if output_path is None:
                if mode == "null":
                    return _resolved(config, mode, options)
                raise ValueError(
                    "LingBot mp4 mode requires output.path in the manifest."
                )
            if mode == "null":
                raise ValueError("LingBot null mode does not write output.path.")
            return _resolved(config, mode, options, output_path=output_path)
        if mode == "webrtc":
            _validate_fields("scenario", options.scenario, _WEBRTC_SCENARIO_FIELDS)
            _validate_fields("output", options.output, _WEBRTC_OUTPUT_FIELDS)
            return _resolved(config, mode, options)
        return None


def _resolved(
    config: RunnerConfig,
    mode: LaunchMode,
    options: LaunchOptions,
    *,
    output_path: object | None = None,
) -> ResolvedLaunch:
    summary: dict[str, object] = {
        "runner": config.runner_name,
        "mode": mode,
        "device": config.device,
    }
    if output_path is not None:
        summary["output_path"] = output_path
    if mode == "webrtc":
        summary["host"] = options.host or options.output.get("host", "0.0.0.0")
        summary["port"] = (
            options.port
            if options.port is not None
            else options.output.get("port", 8080)
        )
    return ResolvedLaunch(
        mode=mode,
        label=f"LingBot {_launch_label(mode)}",
        summary=summary,
        launch=partial(
            _launch,
            config=config,
            mode=mode,
            options=options,
        ),
    )


def _launch(
    *,
    config: RunnerConfig,
    mode: LaunchMode,
    options: LaunchOptions,
) -> object:
    from lingbot.demo.app import launch_from_runner

    if mode not in {"mp4", "null", "webrtc"}:
        raise ValueError(f"Unsupported LingBot launch mode: {mode!r}.")
    return launch_from_runner(
        config=config,
        mode=cast(Literal["mp4", "null", "webrtc"], mode),
        scenario=dict(options.scenario),
        output=dict(options.output),
        host=options.host,
        port=options.port,
        prefer_sw_encoder=options.prefer_sw_encoder,
    )


def _launch_label(mode: LaunchMode) -> str:
    if mode == "mp4":
        return "MP4 replay"
    if mode == "null":
        return "null replay"
    return "WebRTC server"


def _validate_fields(
    section: str,
    values: Mapping[str, object],
    allowed: set[str] | frozenset[str],
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported LingBot {section} fields: {', '.join(unknown)}.")


LAUNCH_CAPABILITY = LingbotLaunchCapability()

__all__ = ["LAUNCH_CAPABILITY", "LingbotLaunchCapability"]
