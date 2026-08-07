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

"""Inference-output handler interface for application-specific results."""

from abc import ABC, abstractmethod
from typing import Any

from flashdreams.runtime.inference_session import InferenceOutput
from pydantic import validate_call


class InferenceOutputHandler(ABC):
    """Interface for consuming output from an inference step."""

    @abstractmethod
    @validate_call
    def __call__(self, inference_output: InferenceOutput) -> Any:
        """Convert inference output into an application-specific result.

        Args:
            inference_output: Output produced by one inference step.

        Returns:
            Handler-specific result.
        """
