# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch capability that routes ``flashdreams-run t2v`` to the T2V app."""

from __future__ import annotations

from functools import partial
from typing import Literal, TypeAlias

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.launch import LaunchMode, LaunchOptions, ResolvedLaunch

from .runner import T2VDemoRunnerConfig

T2VLaunchMode: TypeAlias = Literal["mp4", "null", "webrtc"]


class T2VLaunchCapability:
    """Expose replay and persistent WebRTC modes for the app-owned T2V demo."""

    def supported_modes(
        self, config: RunnerConfig, options: LaunchOptions
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
        t2v_mode = _t2v_mode(mode)
        if t2v_mode is None:
            return None
        t2v_config = _t2v_config(config)
        return ResolvedLaunch(
            mode=t2v_mode,
            label=f"T2V {t2v_mode} launch",
            summary={
                "runner": t2v_config.runner_name,
                "mode": t2v_mode,
                "device": t2v_config.device,
            },
            launch=partial(
                _launch,
                config=t2v_config,
                mode=t2v_mode,
                options=options,
            ),
        )


def _launch(
    *, config: T2VDemoRunnerConfig, mode: T2VLaunchMode, options: LaunchOptions
) -> object:
    from .app import launch_t2v

    return launch_t2v(
        config=config,
        mode=mode,
        host=options.host,
        port=options.port,
        scenario_overrides=dict(options.scenario),
        output_overrides=dict(options.output),
    )


def _t2v_mode(mode: LaunchMode) -> T2VLaunchMode | None:
    if mode == "mp4" or mode == "null" or mode == "webrtc":
        return mode
    return None


def _t2v_config(config: RunnerConfig) -> T2VDemoRunnerConfig:
    if not isinstance(config, T2VDemoRunnerConfig):
        raise TypeError(
            "T2V launch capability requires T2VDemoRunnerConfig, got "
            f"{type(config).__name__}."
        )
    return config


LAUNCH_CAPABILITY = T2VLaunchCapability()

__all__ = ["LAUNCH_CAPABILITY", "T2VLaunchCapability"]
