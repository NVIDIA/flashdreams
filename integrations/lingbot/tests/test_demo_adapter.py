# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    DRIVER_COMMAND,
    CanonicalInputs,
    InferenceConfig,
    InferenceInput,
    StepRequest,
)
from flashdreams.runtime.demo import DemoSpec, LocalWindowOutputSpec
from lingbot.demo.adapter import (
    LingbotDemoAdapter,
    LingbotDriverCommandMapping,
)

pytestmark = pytest.mark.ci_cpu


class _Runtime:
    def __init__(self, config) -> None:
        self.config = config
        self.initialized = False
        self.reset_inputs = []
        self.segments = []
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def reset_for_new_session(self, session_input=None) -> None:
        self.reset_inputs.append(session_input)

    def peek_next_chunk_num_frames(self) -> int:
        return 2

    async def generate_chunk(self, *, segments, frame_times):
        self.segments.append((segments, frame_times))
        return VideoStepResult.from_video_chunk(
            chunk_index=len(self.segments) - 1,
            video_chunk=torch.zeros((1, 1, 2, 3, 4, 6)),
            layout="bvtchw",
        )

    async def close(self) -> None:
        self.closed = True


def test_lingbot_mapping_intentionally_reduces_driver_command_to_wasd() -> None:
    mapping = LingbotDriverCommandMapping()
    mapped = mapping.map_step_inputs(
        canonical_inputs=CanonicalInputs(
            values={
                DRIVER_COMMAND.name: DRIVER_COMMAND.value(
                    {
                        "throttle": 1.0,
                        "brake": 0.0,
                        "steer": -0.5,
                        "stop": False,
                        "reverse": False,
                        "steer_is_direct": False,
                        "manual_control": False,
                    }
                )
            }
        ),
        inference_input=InferenceInput(),
        request=StepRequest(step_index=0),
    )

    assert mapped.step["driver_command"]["throttle"] == 1.0
    assert DRIVER_COMMAND in mapping.mapping_schema.consumes


def test_lingbot_adapter_runs_the_same_standard_session_contract() -> None:
    created: list[_Runtime] = []

    def factory(config):
        runtime = _Runtime(config)
        created.append(runtime)
        return runtime

    adapter = LingbotDemoAdapter(runtime_factory=factory)
    spec = DemoSpec(
        model_id="lingbot",
        input_mode="keyboard-driving",
        config=InferenceConfig(model_id="lingbot", device="cpu"),
        output=LocalWindowOutputSpec(width=6, height=4),
        scenario={"prompt": "drive through a city"},
    )
    prepared = adapter.prepare_session(spec)
    runtime = adapter.create_demo_runtime(spec)
    session = runtime.start_session(prepared.initial_inputs)
    request = session.next_step_request()
    assert request is not None

    result = session.step(
        InferenceInput(
            step={
                "driver_command": {
                    "throttle": 1.0,
                    "brake": 0.0,
                    "steer": 0.5,
                    "stop": False,
                    "reverse": False,
                }
            }
        )
    )

    assert result.frame_count == 2
    assert created[0].segments[0][0][0][2] == frozenset({"w", "a"})
    session.close()
    runtime.close()
    assert created[0].closed
