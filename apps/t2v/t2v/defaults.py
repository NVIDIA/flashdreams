# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What one text-to-video integration contributes to the shared application."""

from dataclasses import dataclass
from typing import Any

from flashdreams.runtime_v2.video_tensor import VideoTensorLayout


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VApplicationDefaults:
    """Defaults one integration supplies to the shared text-to-video application.

    What its model generates when nobody asks for anything in particular. Every
    value can still be overridden on the application's command line.
    """

    pipeline_config: Any
    """Model to load. Owned by the integration, which knows what it is."""

    total_blocks: int
    """Blocks one rollout generates unless a run asks for a different number."""

    pixel_width: int
    """Frame width the model was trained at."""

    pixel_height: int
    """Frame height it was trained at."""

    device: str = "cuda"
    """Device the pipeline is built on."""

    fps: int = 16
    """Rate the generated frames are meant to play at."""

    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    """Layout the pipeline emits."""
