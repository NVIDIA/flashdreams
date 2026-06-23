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

"""CPU checks for Omnidreams runner output post-processing wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omnidreams.config import OMNIDREAMS_RUNNERS
from omnidreams.runner import OmnidreamsRunner

pytestmark = pytest.mark.ci_cpu


def _runner() -> OmnidreamsRunner:
    runner = object.__new__(OmnidreamsRunner)
    runner.config = SimpleNamespace()
    return runner


def test_default_output_keeps_hdmap_and_generated_canvas() -> None:
    runner = _runner()
    condition = torch.full((1, 1, 2, 3, 2, 3), -1.0)
    video = torch.full((1, 1, 2, 3, 2, 3), 1.0)

    canvas, description = runner._prepare_canvas_for_write(
        condition=condition,
        video=video,
    )

    assert description == "HDMap/RGB canvas"
    assert canvas.shape == (2, 4, 3, 3)
    assert torch.equal(canvas[:, :2], torch.full((2, 2, 3, 3), -1.0))
    assert torch.equal(canvas[:, 2:], torch.full((2, 2, 3, 3), 1.0))


def test_omnidreams_runners_process_generated_views_independently() -> None:
    assert OMNIDREAMS_RUNNERS
    for runner_config in OMNIDREAMS_RUNNERS.values():
        assert runner_config.postprocess_output_layout == "bvtchw"
        assert runner_config.postprocess_per_view is True
