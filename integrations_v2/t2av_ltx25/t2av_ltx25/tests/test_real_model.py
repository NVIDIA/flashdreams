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

"""Opt-in smoke test for the pinned LTX 2.5 checkpoint."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from t2av_ltx25 import LTX25Application

pytestmark = pytest.mark.ci_gpu


def test_pinned_checkpoint_writes_joint_audio_video(tmp_path: Path) -> None:
    if os.environ.get("T2AV_LTX25_REAL_MODEL_RUN") != "1":
        pytest.skip("set T2AV_LTX25_REAL_MODEL_RUN=1 for the 70 GB checkpoint smoke")

    app = LTX25Application()
    desc = replace(app.session_desc(), video_width=384, video_height=256)
    video_path = tmp_path / "ltx25-real-smoke.mp4"
    stats_path = tmp_path / "ltx25-real-smoke-stats.json"

    ApplicationRunner(
        app,
        Mp4ClientWindow(video_path),
        metrics_output_sink=MetricsOutputSink(stats_path),
    ).run(
        desc,
        [
            "--prompt",
            "A bronze bell rings once as a silk ribbon moves in the breeze.",
            "--num-frames",
            "9",
            "--seed",
            "42",
        ],
    )

    assert video_path.stat().st_size > 10_000
    assert stats_path.stat().st_size > 0
