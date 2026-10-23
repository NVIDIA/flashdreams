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

"""HandDreams live-calibration state."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LiveCalibrationState:
    """Model-facing pose contract installed by an input source.

    Attributes:
        pose_space: Identifier understood by the HandDreams server.
        epoch: Positive generation attached to every transformed live window.
        control_space: Optional absolute-command contract owned by the source.
    """

    pose_space: str
    epoch: int
    control_space: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pose_space, str) or not self.pose_space:
            raise ValueError("live calibration pose space must be a nonempty string")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int):
            raise ValueError("live calibration epoch must be an integer")
        if self.epoch < 1:
            raise ValueError("live calibration epoch must be positive")
        if self.control_space is not None and (
            not isinstance(self.control_space, str) or not self.control_space
        ):
            raise ValueError(
                "live calibration control space must be a nonempty string or None"
            )
