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

"""OpenXR controller grip and wrist-proxy pose conventions."""

DEFAULT_GRIP_TO_WRIST_OFFSET_M = 0.05
"""Controller-local positive-Z distance from the OpenXR grip to the wrist proxy."""

OPENXR_LOCAL_GRIP_POSE_CONVENTION = "openxr-local-grip-v1"
"""Unmodified OpenXR grip pose in the local reference space."""

OPENXR_STAGE_GRIP_POSE_CONVENTION = "openxr-stage-grip-v1"
"""Unmodified CloudXR grip pose in the server OpenXR STAGE reference space."""

WRIST_PROXY_POSE_CONVENTION = "handdreams-wrist-proxy-v1"
"""Legacy wire identifier for the grip pose with its wrist offset applied."""
