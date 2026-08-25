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

"""OmniDreams binding for the reusable interactive-drive application."""

from __future__ import annotations

from interactive_drive import InteractiveDriveApplication

from flashdreams.api_v2.application import IApplication

from ...config import (
    OMNIDREAMS_APPLICATION_HOOKS,
    resolve_default_interactive_drive_scene,
)


def create_app() -> IApplication:
    """Create the OmniDreams interactive-drive application."""
    return InteractiveDriveApplication(
        hooks=OMNIDREAMS_APPLICATION_HOOKS,
        default_scene_resolver=resolve_default_interactive_drive_scene,
    )


__all__ = ["create_app"]
