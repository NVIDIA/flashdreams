# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch capability that routes ``flashdreams-run t2v`` to the T2V app."""

from __future__ import annotations

from functools import partial
from typing import Literal

from flashdreams.infra.runner import RunnerConfig
from flashdreams.serving.launch import LaunchMode, LaunchOptions, ResolvedLaunch


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
        if mode not in {"mp4", "null", "webrtc"}:
            return None
        return ResolvedLaunch(
            mode=mode,
            label=f"T2V {mode} launch",
            summary={
                "runner": config.runner_name,
                "mode": mode,
                "device": config.device,
            },
            launch=partial(_launch, config=config, mode=mode, options=options),
        )


def _launch(
    *, config: RunnerConfig, mode: LaunchMode, options: LaunchOptions
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


LAUNCH_CAPABILITY = T2VLaunchCapability()

__all__ = ["LAUNCH_CAPABILITY", "T2VLaunchCapability"]
