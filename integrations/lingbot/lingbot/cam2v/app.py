# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot specialization of the shared camera-to-video application."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from cam2v import Cam2VApplication, Cam2VApplicationDefaults, Cam2VConditioning

from flashdreams.api_v2.application import IApplication
from lingbot.config import RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3
from lingbot.input_mapping import load_camera_trace
from lingbot.runtime import replay_inputs_from_mapping

_INSTALL_HINT = "Install the Lingbot plugin: pip install flashdreams-lingbot."


def _resolve_lingbot_conditioning(values: Mapping[str, Any]) -> Cam2VConditioning:
    """Resolve Lingbot example assets into the shared camera contract."""
    replay = replay_inputs_from_mapping(values)
    trace = load_camera_trace(
        camera_poses_path=replay.camera_poses_path,
        camera_intrinsics_path=replay.camera_intrinsics_path,
        pixel_height=replay.pixel_height,
        pixel_width=replay.pixel_width,
        intrinsics_reference_height=480,
        intrinsics_reference_width=832,
        world_scale=replay.world_scale,
    )
    return Cam2VConditioning(
        prompt=replay.prompt,
        first_frame_path=replay.first_frame_path,
        base_intrinsics=trace.intrinsics[0],
        world_scale=trace.world_scale,
    )


class LingbotCam2VApplication(Cam2VApplication):
    """Lingbot World configured through its existing interactive runner config."""

    def __init__(self, *, pipeline_config: Any | None = None) -> None:
        defaults = Cam2VApplicationDefaults.from_runner_config(
            RUNNER_LINGBOT_WORLD_FAST_TAEHV_WINDOW15_SINK3,
            input_resolver=_resolve_lingbot_conditioning,
            install_hint=_INSTALL_HINT,
        )
        if pipeline_config is not None:
            defaults = replace(defaults, pipeline_config=pipeline_config)
        super().__init__(defaults=defaults)


def create_app() -> IApplication:
    """Return a Lingbot camera-to-video application."""
    return LingbotCam2VApplication()


__all__ = ["LingbotCam2VApplication", "create_app"]
