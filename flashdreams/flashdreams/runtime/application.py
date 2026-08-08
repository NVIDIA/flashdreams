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

"""Application configuration and runtime orchestration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from flashdreams.infra.config import InstantiateConfig
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
    InferenceInput,
)
from flashdreams.runtime.input_system import UserInputHandler
from flashdreams.runtime.output_system import InferenceOutputHandler

RuntimeT = TypeVar("RuntimeT", bound=InferenceRuntime)
"""Inference-runtime type owned by the application."""


@dataclass(kw_only=True)
class ApplicationConfig(InstantiateConfig, Generic[RuntimeT]):
    """Configuration for constructing an inference application."""

    _target: type["Application"] = field(default_factory=lambda: Application)

    inference_runtime: InferenceRuntimeConfig[RuntimeT]
    """Configuration used to construct the application runtime."""


class Application(ABC, Generic[RuntimeT]):
    """Own the components required by an inference application.

    Subclasses construct the application-specific input, global-condition, and
    output handlers through initialization hooks called by this base constructor.
    """

    _inference_runtime: RuntimeT
    """Runtime that owns the shared inference pipeline."""

    _user_input_handler: UserInputHandler
    """Handler that produces the next per-step user condition."""

    _inference_global_condition: InferenceGlobalCondition
    """Rollout-wide condition supplied when initializing inference."""

    _global_condition_handler: GlobalConditionHandler
    """Handler that converts application-facing rollout conditions."""

    _inference_output_handler: InferenceOutputHandler
    """Handler that consumes output produced by inference steps."""

    def __init__(
        self,
        config: ApplicationConfig[RuntimeT],
        inference_global_condition: InferenceGlobalCondition,
    ) -> None:
        """Initialize the application from its configuration.

        Args:
            config: Runtime construction configuration.
            inference_global_condition: Initial model-ready rollout condition.

        Raises:
            TypeError: A subclass does not initialize a valid input,
                global-condition, or output handler.
        """
        self._inference_runtime = config.inference_runtime.setup()
        self._inference_global_condition = inference_global_condition

        user_input_handler = self._initialize_user_input_handler(config)
        global_condition_handler = self._initialize_global_condition_handler(config)
        inference_output_handler = self._initialize_inference_output_handler(config)

        if not isinstance(user_input_handler, UserInputHandler):
            raise TypeError(
                f"{type(self).__name__} did not initialize a user input handler"
            )
        if not isinstance(global_condition_handler, GlobalConditionHandler):
            raise TypeError(
                f"{type(self).__name__} did not initialize a global condition handler"
            )
        if not isinstance(inference_output_handler, InferenceOutputHandler):
            raise TypeError(
                f"{type(self).__name__} did not initialize an inference output handler"
            )

        self._user_input_handler = user_input_handler
        self._global_condition_handler = global_condition_handler
        self._inference_output_handler = inference_output_handler

    @abstractmethod
    def _initialize_user_input_handler(
        self, config: ApplicationConfig[RuntimeT]
    ) -> UserInputHandler | None:
        """Construct the application's user-input handler.

        Args:
            config: Application configuration, including any subclass fields.

        Returns:
            Initialized handler, or ``None`` when initialization failed.
        """

    @abstractmethod
    def _initialize_global_condition_handler(
        self, config: ApplicationConfig[RuntimeT]
    ) -> GlobalConditionHandler | None:
        """Construct the application's global-condition handler.

        Args:
            config: Application configuration, including any subclass fields.

        Returns:
            Initialized handler, or ``None`` when initialization failed.
        """

    @abstractmethod
    def _initialize_inference_output_handler(
        self, config: ApplicationConfig[RuntimeT]
    ) -> InferenceOutputHandler | None:
        """Construct the application's inference-output handler.

        Args:
            config: Application configuration, including any subclass fields.

        Returns:
            Initialized handler, or ``None`` when initialization failed.
        """

    def handle_global_condition(
        self, raw_global_condition: RawGlobalCondition
    ) -> InferenceGlobalCondition:
        """Convert and store a raw rollout-wide condition.

        Args:
            raw_global_condition: Application-facing rollout condition.

        Returns:
            Model-ready condition stored for the next application run.

        Raises:
            TypeError: The handler returns a value that is not an inference global
                condition.
        """
        inference_global_condition = self._global_condition_handler(
            raw_global_condition
        )
        if not isinstance(inference_global_condition, InferenceGlobalCondition):
            raise TypeError(
                f"{type(self._global_condition_handler).__name__} did not return an "
                "inference global condition"
            )
        self._inference_global_condition = inference_global_condition
        return inference_global_condition

    def run(self) -> None:
        """Run inference until the user-input handler is exhausted.

        A new inference session is created for the run. The global condition is
        included only in the first inference input because it initializes the
        rollout-wide state for that session.
        """
        inference_session = self._inference_runtime.create_session()
        global_condition: InferenceGlobalCondition | None = (
            self._inference_global_condition
        )

        while True:
            try:
                user_condition = self._user_input_handler()
            except StopIteration:
                return

            inference_input = InferenceInput(
                user_condition=user_condition,
                global_condition=global_condition,
            )
            inference_output = inference_session.step(inference_input)
            self._inference_output_handler(inference_output)
            global_condition = None


__all__ = ["Application", "ApplicationConfig"]
