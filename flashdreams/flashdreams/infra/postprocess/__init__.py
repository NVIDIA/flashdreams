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

"""Video post-processing contracts and tensor utilities."""

from flashdreams.infra.postprocess.base import (
    NoOpVideoPostProcessor,
    NoOpVideoPostProcessorConfig,
    VideoChunk,
    VideoPostprocessChainConfig,
    VideoPostprocessChainSession,
    VideoPostProcessor,
    VideoPostProcessorConfig,
    VideoPostProcessorSession,
    VideoSpec,
    VideoTensorLayout,
    VideoValueRange,
    concatenate_video_chunks,
    from_bvtchw,
    from_minus_one_one,
    infer_video_spec,
    postprocess_video_tensor,
    resize_bvtchw,
    to_bvtchw,
    to_minus_one_one,
)

__all__ = [
    "NoOpVideoPostProcessor",
    "NoOpVideoPostProcessorConfig",
    "VideoChunk",
    "VideoPostProcessor",
    "VideoPostProcessorConfig",
    "VideoPostProcessorSession",
    "VideoPostprocessChainConfig",
    "VideoPostprocessChainSession",
    "VideoSpec",
    "VideoTensorLayout",
    "VideoValueRange",
    "concatenate_video_chunks",
    "from_bvtchw",
    "from_minus_one_one",
    "infer_video_spec",
    "postprocess_video_tensor",
    "resize_bvtchw",
    "to_bvtchw",
    "to_minus_one_one",
]
