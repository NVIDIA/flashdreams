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

"""CPU tests for the LongSana T2V application adapter."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
from longsana.apps.t2v.adapter import (
    LONGSANA_T2V_DEFAULTS,
    LongSanaT2VApplication,
)
from longsana.impl.constants import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    MAX_ROLLOUT_BLOCKS,
)
from t2v.testing import FakeT2VPipelineConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

pytestmark = pytest.mark.ci_cpu


def test_application_advertises_native_one_minute_rollout() -> None:
    """Expose release resolution, cadence, and 26-block long-video default."""
    app = LongSanaT2VApplication(pipeline_config=FakeT2VPipelineConfig())

    description = app.session_desc()

    assert (description.video_width, description.video_height) == (
        DEFAULT_VIDEO_WIDTH,
        DEFAULT_VIDEO_HEIGHT,
    )
    assert description.frames_per_second_for_step == DEFAULT_VIDEO_FPS
    assert LONGSANA_T2V_DEFAULTS.total_blocks == 26


def test_application_rejects_non_native_resolution() -> None:
    """Do not silently sample a resolution outside the validated release path."""
    app = LongSanaT2VApplication(pipeline_config=FakeT2VPipelineConfig())
    app.init(
        [
            "--prompt",
            "A red panda walks through a bamboo forest.",
            "--device",
            "cpu",
            "--total-blocks",
            "1",
        ]
    )
    requested = dataclasses.replace(app.session_desc(), video_width=640)

    with pytest.raises(ValueError, match="requires 832x480"):
        app.create_session(requested)


def test_application_accepts_maximum_rope_bounded_rollout() -> None:
    """Accept the last complete rollout that fits the absolute RoPE table."""
    app = LongSanaT2VApplication(pipeline_config=FakeT2VPipelineConfig())

    app.init(
        [
            "--prompt",
            "A red panda walks through a bamboo forest.",
            "--device",
            "cpu",
            "--total-blocks",
            str(MAX_ROLLOUT_BLOCKS),
        ]
    )


def test_application_rejects_rollout_beyond_rope_table() -> None:
    """Fail before model setup rather than after a long partial generation."""
    app = LongSanaT2VApplication(pipeline_config=FakeT2VPipelineConfig())

    with pytest.raises(ValueError, match="at most 102 blocks"):
        app.init(
            [
                "--prompt",
                "A red panda walks through a bamboo forest.",
                "--device",
                "cpu",
                "--total-blocks",
                str(MAX_ROLLOUT_BLOCKS + 1),
            ]
        )


def test_application_entry_point_is_registered() -> None:
    """Expose the integration through flashdreams-run-v2 discovery."""
    path = Path(__file__).parents[1] / "pyproject.toml"
    with path.open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert project["entry-points"]["flashdreams.applications_v2"] == {
        "t2v-longsana-2b-480p": "longsana.apps.t2v.adapter:create_app"
    }
