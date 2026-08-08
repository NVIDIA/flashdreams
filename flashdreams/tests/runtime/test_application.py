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

"""CPU tests for inference application component ownership."""

from dataclasses import dataclass, field

import pytest
from flashdreams.runtime.application import Application, ApplicationConfig
from flashdreams.runtime.global_condition import (
    GlobalConditionHandler,
    RawGlobalCondition,
)
from flashdreams.runtime.inference_runtime import (
    InferenceRuntime,
    InferenceRuntimeConfig,
)
from flashdreams.runtime.inference_session import (
    InferenceGlobalCondition,
    InferenceOutput,
    InferenceUserCondition,
)
from flashdreams.runtime.input_system import UserInputHandler
from flashdreams.runtime.output_system import InferenceOutputHandler

from .mocks import (
    MockInferenceSession,
    MockStreamInferencePipeline,
    MockStreamInferencePipelineConfig,
)

pytestmark = pytest.mark.ci_cpu


## Component test doubles


class _MockInferenceRuntime(InferenceRuntime[MockInferenceSession]):
    """Runtime test double that skips pipeline construction."""

    def __init__(self, config: InferenceRuntimeConfig) -> None:
        """Retain the runtime config without constructing a pipeline."""
        self.config = config

    def warmup(self) -> None:
        """Complete warmup without model execution."""


class _MockUserInputHandler(UserInputHandler):
    """User-input handler test double."""

    def __call__(self) -> InferenceUserCondition:
        """Return an empty user condition."""
        return InferenceUserCondition()


class _MockInferenceOutputHandler(InferenceOutputHandler):
    """Inference-output handler test double."""

    def __call__(self, inference_output: InferenceOutput) -> None:
        """Consume an inference output without producing a result."""
        del inference_output


class _MockGlobalConditionHandler(GlobalConditionHandler):
    """Global-condition handler test double."""

    def __init__(self) -> None:
        """Initialize the conversion record."""
        self.conditions: list[RawGlobalCondition] = []

    def __call__(
        self, raw_global_condition: RawGlobalCondition
    ) -> InferenceGlobalCondition:
        """Record a raw condition and return a model-ready condition."""
        self.conditions.append(raw_global_condition)
        return InferenceGlobalCondition()


@dataclass(kw_only=True)
class _MockApplicationConfig(ApplicationConfig[_MockInferenceRuntime]):
    """Configuration for the component-ownership application test double."""

    _target: type["_MockApplication"] = field(default_factory=lambda: _MockApplication)


class _MockApplication(Application[_MockInferenceRuntime]):
    """Application test double that constructs no-op handlers."""

    def _initialize_user_input_handler(
        self, config: ApplicationConfig[_MockInferenceRuntime]
    ) -> UserInputHandler:
        """Construct a no-op user-input handler."""
        assert isinstance(config, _MockApplicationConfig)
        return _MockUserInputHandler()

    def _initialize_global_condition_handler(
        self, config: ApplicationConfig[_MockInferenceRuntime]
    ) -> GlobalConditionHandler:
        """Construct a recording global-condition handler."""
        assert isinstance(config, _MockApplicationConfig)
        return _MockGlobalConditionHandler()

    def _initialize_inference_output_handler(
        self, config: ApplicationConfig[_MockInferenceRuntime]
    ) -> InferenceOutputHandler:
        """Construct a no-op inference-output handler."""
        assert isinstance(config, _MockApplicationConfig)
        return _MockInferenceOutputHandler()


## Application ownership


def test_application_privately_owns_inference_components() -> None:
    """Verify construction creates a runtime and retains every component."""
    global_condition = InferenceGlobalCondition()

    runtime_config = InferenceRuntimeConfig(
        _target=_MockInferenceRuntime,
        pipeline=MockStreamInferencePipelineConfig(MockStreamInferencePipeline()),
        session_type=MockInferenceSession,
    )
    config = _MockApplicationConfig(
        inference_runtime=runtime_config,
    )
    application = _MockApplication(config, global_condition)

    assert isinstance(application._inference_runtime, _MockInferenceRuntime)
    assert application._inference_runtime.config is runtime_config
    assert isinstance(application._user_input_handler, _MockUserInputHandler)
    assert application._inference_global_condition is global_condition
    assert isinstance(
        application._global_condition_handler, _MockGlobalConditionHandler
    )
    assert isinstance(
        application._inference_output_handler, _MockInferenceOutputHandler
    )


class _MissingUserInputApplication(_MockApplication):
    """Application test double that fails to construct its input handler."""

    def _initialize_user_input_handler(
        self, config: ApplicationConfig[_MockInferenceRuntime]
    ) -> UserInputHandler | None:
        """Return no user-input handler."""
        del config
        return None


class _MissingInferenceOutputApplication(_MockApplication):
    """Application test double that fails to construct its output handler."""

    def _initialize_inference_output_handler(
        self, config: ApplicationConfig[_MockInferenceRuntime]
    ) -> InferenceOutputHandler | None:
        """Return no inference-output handler."""
        del config
        return None


class _MissingGlobalConditionHandlerApplication(_MockApplication):
    """Application test double that fails to construct its global handler."""

    def _initialize_global_condition_handler(
        self, config: ApplicationConfig[_MockInferenceRuntime]
    ) -> GlobalConditionHandler | None:
        """Return no global-condition handler."""
        del config
        return None


def test_application_rejects_missing_user_input_handler() -> None:
    """Verify construction fails when the child omits its input handler."""
    config = _MockApplicationConfig(
        inference_runtime=InferenceRuntimeConfig(
            _target=_MockInferenceRuntime,
            pipeline=MockStreamInferencePipelineConfig(MockStreamInferencePipeline()),
            session_type=MockInferenceSession,
        ),
    )

    with pytest.raises(TypeError, match="did not initialize a user input handler"):
        _MissingUserInputApplication(config, InferenceGlobalCondition())


def test_application_rejects_missing_inference_output_handler() -> None:
    """Verify construction fails when the child omits its output handler."""
    config = _MockApplicationConfig(
        inference_runtime=InferenceRuntimeConfig(
            _target=_MockInferenceRuntime,
            pipeline=MockStreamInferencePipelineConfig(MockStreamInferencePipeline()),
            session_type=MockInferenceSession,
        ),
    )

    with pytest.raises(
        TypeError, match="did not initialize an inference output handler"
    ):
        _MissingInferenceOutputApplication(config, InferenceGlobalCondition())


def test_application_rejects_missing_global_condition_handler() -> None:
    """Verify construction fails when the child omits its global handler."""
    config = _MockApplicationConfig(
        inference_runtime=InferenceRuntimeConfig(
            _target=_MockInferenceRuntime,
            pipeline=MockStreamInferencePipelineConfig(MockStreamInferencePipeline()),
            session_type=MockInferenceSession,
        ),
    )

    with pytest.raises(
        TypeError, match="did not initialize a global condition handler"
    ):
        _MissingGlobalConditionHandlerApplication(
            config,
            InferenceGlobalCondition(),
        )


def test_application_handles_raw_global_condition() -> None:
    """Verify public conversion replaces the condition used by future runs."""
    config = _MockApplicationConfig(
        inference_runtime=InferenceRuntimeConfig(
            _target=_MockInferenceRuntime,
            pipeline=MockStreamInferencePipelineConfig(MockStreamInferencePipeline()),
            session_type=MockInferenceSession,
        ),
    )
    application = _MockApplication(config, InferenceGlobalCondition())
    raw_global_condition: RawGlobalCondition = {}

    inference_global_condition = application.handle_global_condition(
        raw_global_condition
    )

    handler = application._global_condition_handler
    assert isinstance(handler, _MockGlobalConditionHandler)
    assert handler.conditions == [raw_global_condition]
    assert application._inference_global_condition is inference_global_condition


## Application execution


class _RunInferenceRuntime(InferenceRuntime[MockInferenceSession]):
    """Runtime test double that constructs and returns one session."""

    def __init__(self, config: InferenceRuntimeConfig) -> None:
        """Initialize the session returned by :meth:`create_session`."""
        del config
        self.session = MockInferenceSession()
        self.create_session_calls = 0

    def create_session(self) -> MockInferenceSession:
        """Return the configured session and record the creation request."""
        self.create_session_calls += 1
        return self.session

    def warmup(self) -> None:
        """Complete warmup without model execution."""


class _RunUserInputHandler(UserInputHandler):
    """Finite user-input handler test double."""

    def __init__(self, conditions: list[InferenceUserCondition]) -> None:
        """Initialize with the conditions to return before exhaustion."""
        self.conditions = iter(conditions)
        self.calls = 0

    def __call__(self) -> InferenceUserCondition:
        """Return the next user condition or signal exhaustion."""
        self.calls += 1
        return next(self.conditions)


class _RunInferenceOutputHandler(InferenceOutputHandler):
    """Inference-output handler that records consumed outputs."""

    def __init__(self) -> None:
        """Initialize the consumed output record."""
        self.outputs: list[InferenceOutput] = []

    def __call__(self, inference_output: InferenceOutput) -> None:
        """Record an inference output in call order."""
        self.outputs.append(inference_output)


@dataclass(kw_only=True)
class _RunApplicationConfig(ApplicationConfig[_RunInferenceRuntime]):
    """Configuration for the finite-loop application test double."""

    _target: type["_RunApplication"] = field(default_factory=lambda: _RunApplication)

    user_conditions: list[InferenceUserCondition]
    """Conditions returned by the child-created input handler."""


class _RunApplication(Application[_RunInferenceRuntime]):
    """Application test double that constructs recording handlers."""

    def _initialize_user_input_handler(
        self, config: ApplicationConfig[_RunInferenceRuntime]
    ) -> UserInputHandler:
        """Construct the finite user-input handler."""
        assert isinstance(config, _RunApplicationConfig)
        return _RunUserInputHandler(config.user_conditions)

    def _initialize_global_condition_handler(
        self, config: ApplicationConfig[_RunInferenceRuntime]
    ) -> GlobalConditionHandler:
        """Construct the recording global-condition handler."""
        assert isinstance(config, _RunApplicationConfig)
        return _MockGlobalConditionHandler()

    def _initialize_inference_output_handler(
        self, config: ApplicationConfig[_RunInferenceRuntime]
    ) -> InferenceOutputHandler:
        """Construct the recording inference-output handler."""
        assert isinstance(config, _RunApplicationConfig)
        return _RunInferenceOutputHandler()


def test_application_run_processes_inputs_until_handler_exhaustion() -> None:
    """Verify the application builds inputs and dispatches every output."""
    user_conditions = [InferenceUserCondition(), InferenceUserCondition()]
    global_condition = InferenceGlobalCondition()
    runtime_config = InferenceRuntimeConfig(
        _target=_RunInferenceRuntime,
        pipeline=MockStreamInferencePipelineConfig(MockStreamInferencePipeline()),
        session_type=MockInferenceSession,
    )
    config = _RunApplicationConfig(
        inference_runtime=runtime_config,
        user_conditions=user_conditions,
    )
    application = _RunApplication(config, global_condition)
    runtime = application._inference_runtime
    session = runtime.session
    user_input_handler = application._user_input_handler
    output_handler = application._inference_output_handler

    assert isinstance(user_input_handler, _RunUserInputHandler)
    assert isinstance(output_handler, _RunInferenceOutputHandler)

    application.run()

    # A run owns one stateful session and polls once more to observe exhaustion.
    assert runtime.create_session_calls == 1
    assert user_input_handler.calls == 3

    # The global condition initializes the first step and is omitted thereafter.
    assert len(session.inputs) == 2
    assert session.inputs[0].user_condition is user_conditions[0]
    assert session.inputs[0].global_condition is global_condition
    assert session.inputs[1].user_condition is user_conditions[1]
    assert session.inputs[1].global_condition is None

    # Each inference output is dispatched to the output handler in step order.
    assert output_handler.outputs == session.outputs
