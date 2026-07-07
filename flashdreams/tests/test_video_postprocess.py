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

from flashdreams.infra.postprocess import to_bvtchw
from flashdreams.infra.postprocess.base import (
    from_bvtchw,
)

pytestmark = pytest.mark.ci_cpu


def test_layout_round_trip() -> None:
    video = torch.randint(0, 256, (2, 3, 4, 5, 6), dtype=torch.uint8)
    canonical = to_bvtchw(video, layout="bcthw")
    restored = from_bvtchw(canonical, layout="bcthw")

    assert torch.equal(restored, video)
