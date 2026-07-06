# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner-side construction helpers for streaming video post-processing."""

from __future__ import annotations

from typing import TypeVar

from flashdreams.infra.postprocess.stream import VideoPostprocessStream

RunnerConfigT = TypeVar("RunnerConfigT")


def create_runner_postprocess_stream(
    config: RunnerConfigT,
    *,
    world_size: int,
    is_rank_zero: bool = True,
    fps: float | None = None,
) -> VideoPostprocessStream | None:
    """Create the configured output stream, or ``None`` when disabled."""
    postprocess = getattr(config, "postprocess")
    if not postprocess.is_enabled():
        return None
    postprocess.validate_execution(world_size=world_size)
    if (
        world_size > 1
        and not is_rank_zero
        and not postprocess.requires_all_ranks(world_size=world_size)
    ):
        return None

    output_layout = getattr(config, "postprocess_output_layout")
    if output_layout is None:
        raise ValueError(
            "RunnerConfig.postprocess is enabled but postprocess_output_layout "
            "is not set."
        )

    configured_fps = getattr(config, "postprocess_fps")
    if configured_fps is None:
        configured_fps = fps
    if configured_fps is None:
        configured_fps = getattr(config, "fps", getattr(config, "output_fps", None))

    return VideoPostprocessStream(
        postprocess=postprocess,
        output_layout=output_layout,
        output_value_range=getattr(config, "postprocess_output_value_range"),
        fps=configured_fps,
        per_view=getattr(config, "postprocess_per_view"),
        world_size=world_size,
        profile=bool(
            getattr(getattr(config, "pipeline", None), "enable_sync_and_profile", False)
        ),
    )
