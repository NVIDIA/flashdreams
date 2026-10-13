# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video post-processing contracts and tensor utilities."""

from flashdreams.infra.postprocess.base import (
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
    VideoTensorLayout,
    to_bvtchw,
)
from flashdreams.infra.postprocess.rtx import (
    POSTPROCESS_PRESET_RTX_DEBLUR_ULTRA,
    POSTPROCESS_PRESET_RTX_SUPER_RESOLUTION,
    POSTPROCESS_PRESET_RTX_SUPER_RESOLUTION_ULTRA,
    RTXVideoSuperResolutionPostProcessor,
    RTXVideoSuperResolutionPostProcessorConfig,
    RTXVideoSuperResolutionQuality,
)
from flashdreams.infra.postprocess.stream import (
    VideoPostprocessStepStats,
    VideoPostprocessStream,
    create_runner_postprocess_stream,
    create_video_postprocess_stream,
)

__all__ = [
    "VideoPostprocessStream",
    "VideoPostprocessStepStats",
    "create_runner_postprocess_stream",
    "create_video_postprocess_stream",
    "VideoChunk",
    "VideoPostProcessor",
    "VideoPostProcessorConfig",
    "VideoPostProcessorSession",
    "VideoPostprocessChainConfig",
    "VideoSpec",
    "VideoTensorLayout",
    "to_bvtchw",
    "RTXVideoSuperResolutionPostProcessor",
    "RTXVideoSuperResolutionPostProcessorConfig",
    "RTXVideoSuperResolutionQuality",
    "POSTPROCESS_PRESET_RTX_SUPER_RESOLUTION",
    "POSTPROCESS_PRESET_RTX_SUPER_RESOLUTION_ULTRA",
    "POSTPROCESS_PRESET_RTX_DEBLUR_ULTRA",
]
