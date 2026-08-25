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

"""LingBot binding for the reusable camera-control application."""

from __future__ import annotations

from flashdreams.api_v2.application import IApplication
from action2v import Action2VApplication

from ...config import LINGBOT_APPLICATION_DEFAULTS, LINGBOT_APPLICATION_HOOKS


def create_app() -> IApplication:
    """Create the LingBot camera-control application."""
    return Action2VApplication(
        defaults=LINGBOT_APPLICATION_DEFAULTS,
        hooks=LINGBOT_APPLICATION_HOOKS,
    )


__all__ = ["create_app"]


