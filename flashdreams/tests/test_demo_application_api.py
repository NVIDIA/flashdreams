# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from flashdreams.demo import (
    Application,
    ApplicationSession,
    DemoAdapterApplication,
    FrameOutputSink,
    IOHandler,
    InferenceSessionApplicationAdapter,
    RuntimeOutputSinkFrameAdapter,
)
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InferenceSession,
    InputField,
    InputMapping,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    NullOutputSink,
    NullOutputSpec,
    OutputDecision,
    PreparedScenario,
    SessionInfo,
    UserInputWindow,
)
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepRequirements

pytestmark = pytest.mark.ci_cpu


def test_public_demo_contracts_are_importable() -> None:
    assert Application.__name__ == "Application"
    assert ApplicationSession.__name__ == "ApplicationSession"
    assert IOHandler.__name__ == "IOHandler"
    assert FrameOutputSink.__name__ == "FrameOutputSink"


def test_inference_session_adapter_satisfies_application_session() -> None:
    session = _FakeSession()
    adapter = InferenceSessionApplicationAdapter(session)

    assert isinstance(adapter, ApplicationSession)
    adapter.init()
    requirements = adapter.next_step_requirements()

    assert requirements == StepRequirements(
        step_index=0,
        inference_input_schema=session.inference_input_schema,
    )
    result = adapter.step(InferenceInput(step={"chunk_index": 0}))
    assert result.step_index == 0
    assert adapter.session_info() == SessionInfo(output_layout="thwc")
    adapter.reset()
    adapter.close()
    assert session.initialized
    assert session.reset_called
    assert session.closed


def test_demo_adapter_application_satisfies_application() -> None:
    demo = DemoAdapterApplication(
        adapter=_FakeDemoAdapter(),
        spec=DemoSpec(
            model_id="fake-demo",
            input_mode="replay",
            output=NullOutputSpec(),
        ),
    )

    assert isinstance(demo, Application)
    demo.init(())
    session = demo.create_session()

    assert isinstance(session, ApplicationSession)
    assert session.next_step_requirements() == StepRequirements(
        step_index=0,
        inference_input_schema=_FakeSession.inference_input_schema,
    )
    demo.close()


def test_io_handler_protocol_keeps_input_conversion_outside_io() -> None:
    handler = _FakeIOHandler()

    assert isinstance(handler, IOHandler)
    window = handler.next_window(StepRequirements(step_index=3))

    assert window.start_s == 3.0
    assert window.end_s == 4.0
    assert handler.get_user_input_state("keyboard", "key_w") is False


def test_runtime_output_sink_frame_adapter_satisfies_frame_output_sink() -> None:
    output = NullOutputSink(store_results=True)
    output.open(SessionInfo())
    adapter = RuntimeOutputSinkFrameAdapter(output)

    assert isinstance(adapter, FrameOutputSink)
    adapter.handle_output(
        0.0,
        StepResult(
            step_index=0,
            output="chunk",
            frame_count=1,
            output_window=TimeWindow(start_s=0.0, end_s=1.0),
        ),
    )

    assert output.output_count == 1


class _FakeSession:
    inference_input_schema = InferenceInputSchema(
        step_fields=(InputField(name="chunk_index"),)
    )

    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.reset_called = False

    def init(self) -> None:
        self.initialized = True

    def session_info(self) -> SessionInfo:
        return SessionInfo(output_layout="thwc")

    def next_step_request(self) -> StepRequest | None:
        return StepRequest(
            step_index=0,
            inference_input_schema=self.inference_input_schema,
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        self.inference_input_schema.require_step(inputs)
        return StepResult(
            step_index=0,
            output="chunk",
            frame_count=1,
            output_window=TimeWindow(start_s=0.0, end_s=1.0),
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.reset_called = True

    def close(self) -> None:
        self.closed = True


class _FakeRuntime:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        assert inputs.global_conditioning["prompt"] == "demo"
        return self.session

    def close(self) -> None:
        self.closed = True


class _FakeDemoAdapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name="prompt"),),
        step_fields=(InputField(name="chunk_index"),),
    )
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self) -> None:
        self.runtime = _FakeRuntime()

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("null",)

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        assert config.model_id == self.model_id

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return self.runtime

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        assert spec.model_id == self.model_id
        return PreparedScenario(
            initial_inputs=InferenceInput(global_conditioning={"prompt": "demo"}),
            user_inputs=UserInputs(),
            source_schema=UserInputSchema(),
        )


class _FakeIOHandler:
    def open(self, session_info: SessionInfo) -> None:
        del session_info

    def next_window(self, requirements: StepRequirements) -> UserInputWindow:
        start_s = float(requirements.step_index)
        return UserInputWindow(start_s=start_s, end_s=start_s + 1.0)

    def get_user_input_state(self, modality: str, name: str) -> Any:
        assert modality == "keyboard"
        assert name == "key_w"
        return False

    def begin_generation(self, generation: int) -> None:
        assert generation >= 0

    def emit_chunk(self, result: StepResult) -> OutputDecision:
        assert result.step_index >= 0
        return OutputDecision()

    def should_exit(self) -> bool:
        return False

    def close(self) -> Sequence[OutputArtifact]:
        return ()
