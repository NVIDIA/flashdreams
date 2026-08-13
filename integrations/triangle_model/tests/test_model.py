# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from flashdreams.runtime.demo import DemoSpec, NativeWindowOutputSpec
from triangle_model import create_app

pytestmark = pytest.mark.ci_cpu


def test_triangle_model_emits_moving_frames() -> None:
    application = create_app(
        [
            "--width",
            "16",
            "--height",
            "16",
            "--fps",
            "30",
            "--total-frames",
            "2",
        ]
    )
    spec = DemoSpec(
        model_id=application.model_id,
        input_mode="keyboard-driving",
        output=NativeWindowOutputSpec(
            fps=30,
            video_width=16,
            video_height=16,
        ),
        scenario=application.scenario,
        config=application.config,
    )
    scenario = application.prepare_scenario(spec)
    runtime = application.create_runtime(application.config)
    session = runtime.start_session(scenario.initial_inputs)

    first = session.step(scenario.initial_inputs)
    second = session.step(scenario.initial_inputs)

    assert first.video_chunk.shape == (1, 3, 16, 16)
    assert first.video_chunk.count_nonzero() > 0
    assert not first.video_chunk.equal(second.video_chunk)
    session.close()
    runtime.close()
