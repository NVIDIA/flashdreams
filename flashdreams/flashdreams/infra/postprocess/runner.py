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

"""Runner-side post-processing wiring and distributed gating helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.postprocess.base import (
    VideoPostprocessChainConfig,
    VideoTensorLayout,
    VideoValueRange,
    postprocess_video_tensor,
)

RunnerConfigT = TypeVar("RunnerConfigT")
"""Runner config type accepted by post-processing config helpers."""


def configure_runner_pipeline_postprocess(config: RunnerConfigT) -> RunnerConfigT:
    """Attach runner-level post-processing settings to its pipeline config."""
    postprocess = getattr(config, "postprocess")
    if not postprocess.is_enabled():
        return config

    output_layout = getattr(config, "postprocess_output_layout")
    if output_layout is None:
        raise ValueError(
            "RunnerConfig.postprocess is enabled but postprocess_output_layout "
            "is not set. Set the runner's output layout, or configure "
            "pipeline.postprocess directly."
        )

    fps = getattr(config, "postprocess_fps")
    if fps is None:
        fps = getattr(config, "fps", getattr(config, "output_fps", None))

    return cast(
        RunnerConfigT,
        derive_config(
            config,
            pipeline=dict(
                postprocess=postprocess,
                postprocess_output_layout=output_layout,
                postprocess_output_value_range=getattr(
                    config,
                    "postprocess_output_value_range",
                ),
                postprocess_fps=fps,
                postprocess_per_view=getattr(config, "postprocess_per_view"),
            ),
        ),
    )


def postprocess_requires_all_ranks(
    *,
    world_size: int,
    postprocess: VideoPostprocessChainConfig,
) -> bool:
    """Return whether post-processing must run on every distributed rank."""
    return world_size > 1 and postprocess.uses_context_parallelism()


def validate_runner_postprocess_for_world_size(
    *,
    world_size: int,
    postprocess: VideoPostprocessChainConfig,
) -> None:
    """Validate runner post-processing against the distributed world size."""
    if world_size <= 1 or not postprocess.is_enabled():
        return
    if postprocess.uses_context_parallelism():
        return
    for processor in postprocess.resolved_processors():
        attention_mode = getattr(processor, "attention_mode", None)
        if attention_mode == "sparse":
            raise ValueError(
                "FlashVSR sparse post-processing does not support multi-GPU "
                "execution. Use --postprocess.preset flashvsr-v1.1-full-attn "
                "for context parallelism, or run without torchrun."
            )


def apply_runner_output_postprocess(
    tensor: Tensor,
    *,
    layout: VideoTensorLayout,
    value_range: VideoValueRange,
    fps: float | None,
    postprocess: VideoPostprocessChainConfig,
    world_size: int,
    is_rank_zero: bool,
    postprocess_video: Callable[..., Tensor] = postprocess_video_tensor,
) -> Tensor | None:
    """Apply in-memory runner post-processing with distributed rank gating."""
    if not postprocess.is_enabled():
        return tensor

    validate_runner_postprocess_for_world_size(
        world_size=world_size,
        postprocess=postprocess,
    )

    if postprocess_requires_all_ranks(
        world_size=world_size,
        postprocess=postprocess,
    ):
        output = postprocess_video(
            tensor,
            layout=layout,
            value_range=value_range,
            fps=fps,
        )
        return output.cpu() if is_rank_zero else None

    if is_rank_zero:
        return postprocess_video(
            tensor,
            layout=layout,
            value_range=value_range,
            fps=fps,
        ).cpu()

    return None
