# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import pytest
from flashdreams.runtime import (
    CanonicalInputs,
    CanonicalInputSchema,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceOutputSchema,
    InputCanonicalizer,
    InputField,
    InputMappingSchema,
    InteractiveInferenceWorker,
    InteractiveSessionEnded,
    InteractiveSessionJob,
    InteractiveStep,
    StepRequest,
    StepResult,
    UserInputs,
    UserInputSchema,
)

pytestmark = pytest.mark.ci_cpu


class _Mapping:
    mapping_schema = InputMappingSchema(
        name="interactive-test",
        produces_global_conditioning=(InputField(name="prompt"),),
        produces_step=(InputField(name="chunk_index"),),
    )

    def validate(self, **_: object) -> None:
        return

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        del canonical_inputs
        return inference_input

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        del canonical_inputs
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step={"chunk_index": request.step_index},
            metadata=inference_input.metadata,
        )


class _Adapter:
    model_id = "interactive-test"
    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name="prompt"),),
        step_fields=(InputField(name="chunk_index"),),
    )
    inference_output_schema = InferenceOutputSchema(
        modality="text/plain",
        python_type=str,
    )
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self, *, steps: int = 2, step_delay_s: float = 0.0) -> None:
        self.steps = steps
        self.step_delay_s = step_delay_s
        self.runtime: _Runtime | None = None
        self.create_count = 0

    def default_input_mapping(self) -> _Mapping:
        return _Mapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(config.model_id)

    def create_runtime(self, config: InferenceConfig) -> _Runtime:
        self.create_count += 1
        self.runtime = _Runtime(
            config=config,
            schema=self.inference_input_schema,
            steps=self.steps,
            step_delay_s=self.step_delay_s,
        )
        return self.runtime


class _Runtime:
    def __init__(
        self,
        *,
        config: InferenceConfig,
        schema: InferenceInputSchema,
        steps: int,
        step_delay_s: float,
    ) -> None:
        self.config = config
        self.schema = schema
        self.steps = steps
        self.step_delay_s = step_delay_s
        self.created_thread = threading.get_ident()
        self.closed_thread: int | None = None

    def start_session(self, inputs: InferenceInput) -> _Session:
        self.schema.require_global_conditioning(inputs)
        return _Session(
            schema=self.schema,
            steps=self.steps,
            step_delay_s=self.step_delay_s,
        )

    def close(self) -> None:
        self.closed_thread = threading.get_ident()


class _Session:
    def __init__(
        self,
        *,
        schema: InferenceInputSchema,
        steps: int,
        step_delay_s: float,
    ) -> None:
        self.schema = schema
        self.steps = steps
        self.step_delay_s = step_delay_s
        self.index = 0

    def next_step_request(self) -> StepRequest | None:
        if self.index >= self.steps:
            return None
        return StepRequest(step_index=self.index)

    def step(self, inputs: InferenceInput) -> StepResult:
        self.schema.require_step(inputs)
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        result = StepResult(step_index=self.index, output=f"step-{self.index}")
        self.index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.index = 0

    def close(self) -> None:
        return


def _job(session_id: str) -> InteractiveSessionJob:
    return InteractiveSessionJob(
        session_id=session_id,
        mapping=_Mapping(),
        canonicalizer=InputCanonicalizer(),
        source_schema=UserInputSchema(),
        user_inputs=UserInputs(),
        initial_inputs=InferenceInput(global_conditioning={"prompt": "go"}),
    )


def _collect_until_end(
    worker: InteractiveInferenceWorker,
) -> tuple[list[InteractiveStep], InteractiveSessionEnded]:
    steps: list[InteractiveStep] = []
    while True:
        event = worker.get_event(timeout_s=1.0)
        assert event is not None
        if isinstance(event, InteractiveStep):
            steps.append(event)
        else:
            return steps, event


def test_worker_streams_results_and_keeps_runtime_thread_affine() -> None:
    adapter = _Adapter()
    main_thread = threading.get_ident()
    worker = InteractiveInferenceWorker(
        adapter=adapter,
        config=InferenceConfig(model_id=adapter.model_id),
    )
    worker.start()
    worker.submit(_job("scene-a"))

    steps, ended = _collect_until_end(worker)
    worker.close()

    assert [step.result.output for step in steps] == ["step-0", "step-1"]
    assert ended.error is None
    assert adapter.runtime is not None
    assert adapter.runtime.created_thread != main_thread
    assert adapter.runtime.closed_thread == adapter.runtime.created_thread


def test_worker_reuses_one_runtime_for_sequential_sessions() -> None:
    adapter = _Adapter()
    worker = InteractiveInferenceWorker(
        adapter=adapter,
        config=InferenceConfig(model_id=adapter.model_id),
    )
    worker.start()

    worker.submit(_job("scene-a"))
    _collect_until_end(worker)
    worker.submit(_job("scene-b"))
    _collect_until_end(worker)
    worker.close()

    assert adapter.create_count == 1


def test_stop_request_ends_after_the_current_blocking_step() -> None:
    adapter = _Adapter(steps=20, step_delay_s=0.01)
    worker = InteractiveInferenceWorker(
        adapter=adapter,
        config=InferenceConfig(model_id=adapter.model_id),
    )
    worker.start()
    worker.submit(_job("scene-a"))

    first = worker.get_event(timeout_s=1.0)
    assert isinstance(first, InteractiveStep)
    worker.stop_session()
    remaining, ended = _collect_until_end(worker)
    worker.close()

    assert ended.stopped
    assert len(remaining) <= 1
