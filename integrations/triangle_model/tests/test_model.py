# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest
from flashdreams.runtime import InferenceConfig
from flashdreams.runtime.demo import DemoSpec, NativeWindowOutputSpec
from triangle_app import TriangleScenario
from triangle_model import MODEL_ID, TriangleModel

pytestmark = pytest.mark.ci_cpu


def test_triangle_model_emits_moving_frames() -> None:
    adapter = TriangleModel()
    spec = DemoSpec(
        model_id=MODEL_ID,
        input_mode="keyboard-driving",
        output=NativeWindowOutputSpec(
            fps=30,
            video_width=16,
            video_height=16,
        ),
        scenario=TriangleScenario(
            width=16,
            height=16,
            fps=30,
            total_frames=2,
        ),
        config=InferenceConfig(model_id=MODEL_ID, device="cpu"),
    )
    scenario = adapter.prepare_scenario(spec)
    runtime = adapter.create_runtime(cast(InferenceConfig, spec.config))
    session = runtime.start_session(scenario.initial_inputs)

    first = session.step(scenario.initial_inputs)
    second = session.step(scenario.initial_inputs)

    assert first.video_chunk.shape == (1, 3, 16, 16)
    assert first.video_chunk.count_nonzero() > 0
    assert not first.video_chunk.equal(second.video_chunk)
    session.close()
    runtime.close()
