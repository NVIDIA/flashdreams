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

"""User-input contracts and handler interface for inference conditioning."""

from abc import ABC, abstractmethod
from typing import Annotated

from flashdreams.runtime.inference_session import InferenceUserCondition
from pydantic import ConfigDict, Field, validate_call, with_config
from typing_extensions import TypedDict


@with_config(ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class RawUserInput(TypedDict):
    """Base typed dictionary for device-specific user input."""

    timestamp: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    """Event timestamp in seconds on the input source's monotonic clock."""


@with_config(ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class CanonicalizedUserInput(TypedDict):
    """Base typed dictionary for device-independent user intent."""


class UserInputHandler(ABC):
    """Interface for producing inference conditioning from user input."""

    @abstractmethod
    @validate_call
    def __call__(self) -> InferenceUserCondition:
        """Return a model-ready per-step condition.

        Returns:
            Model-ready condition for one inference step.
        """
