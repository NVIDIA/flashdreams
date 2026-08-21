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

"""Cosmos Predict2 text-to-video application factory."""

from t2v import (
    T2VApplication,
    T2VApplicationDefaults,
    T2VApplicationSession,
)

from cosmos_predict2.config import RUNNER_COSMOS2_T2V_2B_720P
from flashdreams.demo import IFlashDreamsApplication


class CosmosPredict2T2VApplication(T2VApplication):
    """Cosmos Predict2 text-to-video application."""

    session_type = T2VApplicationSession

    def __init__(self) -> None:
        super().__init__(
            defaults=T2VApplicationDefaults.from_runner_config(
                RUNNER_COSMOS2_T2V_2B_720P
            )
        )


def create_app() -> IFlashDreamsApplication:
    """Create the Cosmos Predict2 text-to-video application."""
    return CosmosPredict2T2VApplication()


__all__ = [
    "CosmosPredict2T2VApplication",
    "create_app",
]
