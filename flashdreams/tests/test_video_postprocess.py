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

"""CPU tests for generic video post-processing utilities."""

from __future__ import annotations

import pytest
import torch

from flashdreams.infra.postprocess import (
    VideoPostprocessChainConfig,
    from_bvtchw,
    postprocess_video_tensor,
    to_bvtchw,
    to_minus_one_one,
)

pytestmark = pytest.mark.ci_cpu


def test_noop_chain_returns_same_tchw_tensor() -> None:
    video = torch.linspace(-1.0, 1.0, steps=3 * 3 * 4 * 5).reshape(3, 3, 4, 5)

    result = postprocess_video_tensor(
        video,
        layout="tchw",
        value_range="minus_one_one",
        postprocess=VideoPostprocessChainConfig(),
        fps=24,
    )

    assert torch.equal(result, video)


def test_layout_and_uint8_round_trip() -> None:
    video = torch.randint(0, 256, (2, 4, 5, 3), dtype=torch.uint8)
    canonical = to_bvtchw(video, layout="thwc")
    restored = from_bvtchw(canonical, layout="thwc")

    assert torch.equal(restored, video)
    normalized = to_minus_one_one(video, value_range="uint8")
    assert normalized.dtype == torch.float32
    assert normalized.min() >= -1.0
    assert normalized.max() <= 1.0
