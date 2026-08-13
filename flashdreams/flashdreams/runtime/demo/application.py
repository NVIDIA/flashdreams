# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application package contract."""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, runtime_checkable

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime import InferenceConfig

from .spec import DemoAdapter

ApplicationMode: TypeAlias = Literal["mp4", "null", "webrtc", "local-window"]


@runtime_checkable
class FlashDreamsApplication(DemoAdapter, Protocol):
    application_name: str
    description: str
    config: InferenceConfig
    scenario: object
    fps: int
    video_width: int
    video_height: int
    output_layout: VideoTensorLayout
    default_mode: ApplicationMode
    title: str | None
    supported_control_keys: frozenset[str]


__all__ = ["ApplicationMode", "FlashDreamsApplication"]
