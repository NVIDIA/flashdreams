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

"""CPU tests for the builtin video-output application."""

from pathlib import Path

import pytest
from flashdreams.runtime.application import Application, ApplicationConfig
from flashdreams.runtime.builtin.application.video_output_application import (
    VideoOutputApplication,
    VideoOutputApplicationConfig,
)
from flashdreams.runtime.builtin.inference_output.handler.video_output_handler import (
    VideoOutputHandler,
)
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
    InferenceUserCondition,
)
from flashdreams.runtime.input_system import UserInputHandler

from .mocks import (
    MockInferenceSession,
    MockStreamInferencePipeline,
    MockStreamInferencePipelineConfig,
)

pytestmark = pytest.mark.ci_cpu


class _MockInferenceRuntime(InferenceRuntime[MockInferenceSession]):
    """Runtime test double that skips pipeline construction."""

    def __init__(self, config: InferenceRuntimeConfig) -> None:
        """Retain the runtime configuration."""
        self.config = config

    def warmup(self) -> None:
        """Complete warmup without model execution."""


class _MockUserInputHandler(UserInputHandler):
    """User-input handler test double."""

    def __call__(self) -> InferenceUserCondition:
        """Return an empty user condition."""
        return InferenceUserCondition()


class _MockGlobalConditionHandler(GlobalConditionHandler):
    """Global-condition handler test double."""

    def __call__(
        self, raw_global_condition: RawGlobalCondition
    ) -> InferenceGlobalCondition:
        """Return an empty inference global condition."""
        del raw_global_condition
        return InferenceGlobalCondition()


class _TestVideoOutputApplication(VideoOutputApplication[_MockInferenceRuntime]):
    """Concrete video-output application test double."""

    def _initialize_user_input_handler(
        self, config: ApplicationConfig[_MockInferenceRuntime]
    ) -> UserInputHandler:
        """Construct the application user-input handler."""
        del config
        return _MockUserInputHandler()

    def _initialize_global_condition_handler(
        self, config: ApplicationConfig[_MockInferenceRuntime]
    ) -> GlobalConditionHandler:
        """Construct the global-condition handler."""
        del config
        return _MockGlobalConditionHandler()


def test_video_output_application_initializes_video_handler(tmp_path: Path) -> None:
    """Verify construction binds the artifact path to a video output handler."""
    runtime_config = InferenceRuntimeConfig(
        _target=_MockInferenceRuntime,
        pipeline=MockStreamInferencePipelineConfig(MockStreamInferencePipeline()),
        session_type=MockInferenceSession,
    )
    artifact_path = tmp_path / "generated.mp4"
    config = VideoOutputApplicationConfig(
        inference_runtime=runtime_config,
        artifact_path=artifact_path,
    )

    application = _TestVideoOutputApplication(
        config,
        InferenceGlobalCondition(),
    )

    # The base constructor obtains each handler through its child initialization hook.
    assert isinstance(application._user_input_handler, _MockUserInputHandler)
    assert isinstance(application._inference_output_handler, VideoOutputHandler)
    assert application._inference_output_handler.artifact_path == artifact_path


def test_video_output_application_finishes_video_handler_after_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify a completed application run finishes its video output handler."""
    lifecycle: list[str] = []

    def run(_application: Application) -> None:
        lifecycle.append("run")

    def finish(handler: VideoOutputHandler) -> Path:
        lifecycle.append("finish")
        return handler.artifact_path

    monkeypatch.setattr(Application, "run", run)
    monkeypatch.setattr(VideoOutputHandler, "finish", finish)
    config = VideoOutputApplicationConfig(
        inference_runtime=InferenceRuntimeConfig(
            _target=_MockInferenceRuntime,
            pipeline=MockStreamInferencePipelineConfig(MockStreamInferencePipeline()),
            session_type=MockInferenceSession,
        ),
        artifact_path=tmp_path / "generated.mp4",
    )
    application = _TestVideoOutputApplication(config, InferenceGlobalCondition())

    application.run()

    assert lifecycle == ["run", "finish"]
