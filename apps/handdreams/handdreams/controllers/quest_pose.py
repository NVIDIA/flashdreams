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

"""Quest grip-pose conventions and Isaac Teleop's wrist-offset default."""

from flashdreams_controller_input.pose import (
    DEFAULT_GRIP_TO_WRIST_OFFSET_M as DEFAULT_GRIP_TO_WRIST_OFFSET_M,
)
from flashdreams_controller_input.pose import (
    OPENXR_LOCAL_GRIP_POSE_CONVENTION as OPENXR_LOCAL_GRIP_POSE_CONVENTION,
)
from flashdreams_controller_input.pose import (
    WRIST_PROXY_POSE_CONVENTION,
)

HANDDREAMS_WRIST_PROXY_POSE_CONVENTION = WRIST_PROXY_POSE_CONVENTION
"""Canonical backend pose after applying the configured grip-to-wrist offset."""
